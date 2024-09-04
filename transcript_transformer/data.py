import os
import traceback
import logging
from copy import deepcopy
import shutil
import time
import numpy as np

from scipy import sparse
from tqdm import tqdm
import biobear as bb
import polars as pl

import h5py
import h5max
import pyfaidx
from gtfparse import read_gtf

from .util_functions import vec2DNA, construct_prot


def co_to_idx(start, end):
    return start - 1, end


def slice_gen(
    seq,
    start,
    end,
    strand,
    co=True,
    to_vec=True,
    seq_dict={"A": 0, "T": 1, "C": 2, "G": 3, "N": 4},
    comp_dict={0: 1, 1: 0, 2: 3, 3: 2, 4: 4},
):
    """get sequence following gtf-coordinate system"""
    if co:
        start, end = co_to_idx(start, end)
    sl = seq[start:end].seq

    if to_vec:
        sl = list(map(lambda x: seq_dict[x.upper()], sl))

    if strand in ["-", -1, False]:
        if comp_dict is not None:
            sl = list(map(lambda x: comp_dict[x], sl))[::-1]
        else:
            sl = sl[::-1]

    return np.array(sl)


def process_seq_data(h5_path, gtf_path, fa_path, backup_path, backup=True):
    pulled = False
    if not backup_path:
        backup_path = os.path.splitext(gtf_path)[0] + ".h5"
    if os.path.abspath(backup_path) == os.path.abspath(os.path.dirname(h5_path)):
        print(f"!-> Backup path identical to h5 output path, disabling backup...")
        backup = False
    elif not os.path.isfile(h5_path) and os.path.isfile(backup_path):
        print(f"--> Processed assembly data restored ({backup_path})")
        shutil.copy(backup_path, h5_path)
        pulled = True
    if os.path.isfile(h5_path):
        f = h5py.File(h5_path, "r")
        if "transcript" in f.keys():
            print(
                "--> Parsed transcriptome directory found, "
                "assembly information can not be re-processed (for existing h5 files)."
            )
        f.close()
    else:
        data_dict = parse_transcriptome(gtf_path, fa_path)
        no_handle = True
        max_wait = 900
        waited = 0
        while no_handle and (waited < max_wait):
            try:
                f = h5py.File(h5_path, "a")
                no_handle = False
            except Exception as e:
                if waited < max_wait:
                    time.sleep(120)
                    waited += 120
        if not no_handle:
            try:
                f = save_transcriptome_to_h5(f, data_dict)
                f.close()
                if backup and (not pulled):
                    shutil.copy(h5_path, backup_path)
            except Exception as e:
                logging.error(traceback.format_exc())
                print("Failed to update h5 database, which might be corrupted")
        else:
            print("Could not open h5 database, suspending...")


def process_ribo_data(
    h5_path, ribo_paths, overwrite=False, parallel=False, low_memory=False
):
    f = h5py.File(h5_path, "r")
    tr_ids = pl.from_numpy(np.array(f["transcript/id"])).to_series().cast(pl.Utf8)
    tr_lens = pl.from_numpy(np.array(f["transcript/tr_len"])).to_series()
    header_dict = {2: "tr_ID", 3: "pos", 9: "read"}
    ribo_to_parse = deepcopy(ribo_paths)
    for experiment, path in ribo_paths.items():
        cond_1 = (
            parallel
            and (not overwrite)
            and (os.path.isfile(h5_path.split(".h5")[0] + f"_{experiment}.h5"))
        )
        cond_2 = not (parallel or overwrite) and (
            f"transcript/riboseq/{experiment}" in f.keys()
        )
        if cond_1 or cond_2:
            print(
                f"--> {experiment} in h5, omitting..."
                "(use --overwrite for overwriting existing riboseq data)"
            )
            ribo_to_parse.pop(experiment)
    f.close()
    for experiment, path in ribo_to_parse.items():
        print(f"Loading in {experiment}...")
        _, file_ext = os.path.splitext(path)
        if file_ext == ".sam":
            df = pl.read_csv(
                path,
                has_header=False,
                comment_prefix="@",
                columns=[2, 3, 9],
                dtypes=[pl.Utf8, pl.Int32, pl.Utf8],
                separator="\t",
                low_memory=low_memory,
            )
        elif file_ext == ".bam":
            s = f"CREATE EXTERNAL TABLE test STORED AS BAM LOCATION '{path}'"
            ctx = bb.connect()
            ctx.sql(s)
            df = ctx.sql("SELECT reference, start, sequence FROM test").to_polars()
        else:
            raise TypeError(f"file extension {file_ext} not supported")
        df.columns = list(header_dict.values())
        # TODO implement option to run custom read lens
        read_lens = np.arange(20, 41)
        riboseq_data = parse_ribo_reads(df, read_lens, tr_ids, tr_lens)
        try:
            print("Saving data...")
            if not parallel:
                f = h5py.File(h5_path, "a")
            else:
                f = h5py.File(h5_path.split(".h5")[0] + f"_{experiment}.h5", "w")
                f.create_group("transcript")
            if "riboseq" not in f["transcript"].keys():
                f["transcript"].create_group("riboseq")
            if experiment in f["transcript/riboseq"].keys():
                del f[f"transcript/riboseq/{experiment}"]
            f["transcript/riboseq"].create_group(experiment)
            exp_grp = f[f"transcript/riboseq/{experiment}"].create_group("5")
            h5max.store_sparse(exp_grp, riboseq_data, format="csr")
            num_reads = [s.sum() for s in riboseq_data]
            exp_grp.create_dataset("num_reads", data=np.array(num_reads).astype(int))
            exp_grp.create_dataset("metadata", data=read_lens)
            f.close()
        except Exception as error:
            print(error)
            del f[f"transcript/riboseq/{experiment}"]


def save_transcriptome_to_h5(f, data_dict):
    print("Save data in hdf5 files...")
    dt8 = h5py.vlen_dtype(np.dtype("int8"))
    dt = h5py.vlen_dtype(np.dtype("int"))
    grp = f.create_group("transcript")
    for key, array in data_dict.items():
        if key in [
            "id",
            "contig",
            "gene_id",
            "gene_name",
            "strand",
            "biotype",
            "tag",
            "support_lvl",
            "canonical_protein_id",
        ]:
            if key != "id":
                array = [a if a != None else "" for a in array]
            grp.create_dataset(
                key, data=array, dtype=f"<S{max(1,max([len(s) for s in array]))}"
            )
        elif key in ["seq", "tis"]:
            grp.create_dataset(key, data=array, dtype=dt8)
        elif key in ["exon_idxs", "exon_coords", "CDS_idxs", "CDS_coords"]:
            grp.create_dataset(key, data=np.array(array, dtype=object), dtype=dt)
        else:
            grp.create_dataset(key, data=array)

    return f


def parse_transcriptome(gtf_path, fa_path):
    print("Loading assembly data...")
    genome = pyfaidx.Fasta(fa_path)
    contig_list = pl.Series(genome.keys())
    gtf = read_gtf(gtf_path, result_type="polars")
    gtf = gtf.with_columns(pl.col("exon_number").cast(pl.Int32, strict=False))
    headers = {
        "transcript_id": "id",
        "gene_id": "gene_id",
        "gene_name": "gene_name",
        "strand": "strand",
        "transcript_biotype": "biotype",
        "tag": "tag",
        "transcript_support_level": "support_lvl",
    }

    print("Extracting transcripts and metadata...")
    data_dict = {
        "id": [],
        "seq": [],
        "tis": [],
        "gene_id": [],
        "gene_name": [],
        "contig": [],
        "strand": [],
        "biotype": [],
        "tag": [],
        "support_lvl": [],
        "canonical_TIS_exon_idx": [],
        "exon_idxs": [],
        "exon_coords": [],
        "CDS_idxs": [],
        "CDS_coords": [],
        "canonical_TIS_idx": [],
        "canonical_TIS_coord": [],
        "canonical_TTS_idx": [],
        "canonical_TTS_coord": [],
        "tr_len": [],
        "canonical_protein_id": [],
        "canonical_protein_seq": [],
    }
    assert "transcript_id" in gtf.columns, "transcript_id column missing in gtf file"
    assert "strand" in gtf.columns, "strand column missing in gtf file"
    for key, _ in headers.items():
        if key not in gtf.columns:
            gtf = gtf.with_columns(pl.lit("").alias(key))

    for contig in contig_list:
        print(f"{contig}...")
        gtf_set = gtf.filter(pl.col("seqname") == str(contig))
        tr_set = gtf_set["transcript_id"].unique()
        tr_set = tr_set.filter(tr_set != "")
        for i, id in tqdm(enumerate(tr_set), total=len(tr_set)):
            # obtain transcript information
            gtf_tr = gtf_set.filter(pl.col("transcript_id") == id).sort(
                by="exon_number"
            )
            keys, values = (
                (
                    gtf_tr.filter(pl.col("feature") == "transcript").select(
                        list(headers.keys())
                    )
                )
                .melt()
                .to_dict()
                .values()
            )
            for k, i in zip(list(headers.values()), values):
                data_dict[k].append(i)

            # obtain and sort exon information (strings have wrong sortin (e.g. 10, 11, 2, 3, ...))
            exons = gtf_tr.filter(pl.col("feature") == "exon")
            exon_lens = (abs(exons["start"] - exons["end"]) + 1).to_numpy()
            cum_exon_lens = np.insert(np.cumsum(exon_lens), 0, 0)
            cdss = gtf_tr.filter(pl.col("feature") == "CDS")
            if len(cdss) > 0:
                cds_lens = (abs(cdss["start"] - cdss["end"]) + 1).to_numpy()
                cum_cds_lens = np.insert(np.cumsum(cds_lens), 0, 0)
                cds_idxs = np.vstack((cum_cds_lens[:-1], cum_cds_lens[1:])).T.ravel()

            strand_is_pos = (exons["strand"] == "+").any()
            data_dict["tr_len"].append(exon_lens.sum())

            if len(exons) == 0:
                print(
                    "WARNING: No exons found for transcript. This should not happen. Please ensure"
                    "exons are marked with the correct transcript id"
                )

            # obtain TISs, select first in case of split (intron) start codon
            # TODO: when multiple TISs are supported, code needs update
            start_codon = (
                gtf_tr.filter(pl.col("feature") == "start_codon").slice(0, 1).to_dicts()
            )
            stop_codon = (
                gtf_tr.filter(pl.col("feature") == "stop_codon").slice(0, 1).to_dicts()
            )

            target_seq = np.full(exon_lens.sum(), False)

            if len(start_codon) > 0:
                # use as index for sorted dfs
                start_codon = start_codon[0]
                exon_i = start_codon["exon_number"] - 1
                exon = exons[exon_i].to_dicts()[0]
                if strand_is_pos:
                    tis = start_codon["start"]
                    tis_idx = cum_exon_lens[exon_i] + tis - exon["start"]
                    if len(stop_codon) > 0:
                        tts = stop_codon[0]["start"]
                    else:
                        tts = exons[-1].to_dicts()[0]["end"]
                else:
                    tis = start_codon["end"]
                    tis_idx = cum_exon_lens[exon_i] + exon["end"] - tis
                    if len(stop_codon) > 0:
                        tts = stop_codon[0]["end"]
                    else:
                        tts = exons[-1].to_dicts()[0]["start"]

                target_seq[tis_idx] = 1
                data_dict["canonical_TIS_exon_idx"].append(exon_i)
                data_dict["canonical_TIS_idx"].append(tis_idx)
                data_dict["canonical_TTS_idx"].append(tis_idx + sum(cds_lens))
                # remove potential empty entries from protein_id ("")
                prot_ids = (
                    gtf_tr["protein_id"].to_frame().filter(pl.all() != "").to_series()
                )
                prot_id = prot_ids.unique(maintain_order=True)[0]
                data_dict["canonical_protein_id"].append(prot_id)
                data_dict["canonical_TIS_coord"].append(tis)
                data_dict["canonical_TTS_coord"].append(tts)

            else:
                data_dict["canonical_TIS_exon_idx"].append(-1)
                data_dict["canonical_TIS_idx"].append(-1)
                data_dict["canonical_TTS_idx"].append(-1)
                data_dict["canonical_protein_id"].append("")
                data_dict["canonical_TIS_coord"].append(-1)
                data_dict["canonical_TTS_coord"].append(-1)
                # some transcripts have CDSs but no start codons...

            if len(cdss) > 0:
                if strand_is_pos:
                    exon_i = cdss[0, "exon_number"] - 1
                    exon_shift = cum_exon_lens[exon_i]
                    cds_offset = exon_shift + cdss[0, "start"] - exons[exon_i, "start"]
                else:
                    exon_i = cdss[-1, "exon_number"] - 1
                    exon_shift = cum_exon_lens[exon_i]
                    cds_offset = exon_shift + exons[exon_i, "end"] - cdss[-1, "end"]
                data_dict["CDS_idxs"].append(cds_idxs + cds_offset)
                data_dict["CDS_coords"].append(
                    cdss[:, ["start", "end"]].transpose().melt()["value"].to_numpy()
                )
            else:
                data_dict["CDS_idxs"].append(np.empty(0, dtype=int))
                data_dict["CDS_coords"].append(np.empty(0, dtype=int))

            exon_coords = []
            exon_seqs = []
            for exon_i, exon in enumerate(exons.iter_rows(named=True)):
                # get sequence
                exon_seq = slice_gen(
                    genome[contig],
                    exon["start"],
                    exon["end"],
                    exon["strand"],
                    to_vec=True,
                ).astype(np.int16)

                exon_coords.append(exon["start"])
                exon_coords.append(exon["end"])
                exon_seqs.append(exon_seq)
            exon_idxs = np.vstack((cum_exon_lens[:-1], cum_exon_lens[1:])).T.ravel()
            seq = np.concatenate(exon_seqs)

            if len(start_codon) > 0:
                DNA_frag = vec2DNA(seq[data_dict["canonical_TIS_idx"][-1] :])
                prot, _, _ = construct_prot(DNA_frag)
            else:
                prot = ""

            data_dict["exon_idxs"].append(exon_idxs)
            data_dict["exon_coords"].append(np.array(exon_coords))
            data_dict["seq"].append(seq)
            data_dict["tis"].append(target_seq)
            data_dict["canonical_protein_seq"].append(prot)
            data_dict["contig"].append(contig)

    return data_dict


def parse_ribo_reads(df, read_lens, f_ids, f_lens):
    num_read_lens = len(read_lens)
    read_len_dict = {read_len: i for i, read_len in enumerate(read_lens)}
    print("Filtering on read lens...")
    df = df.with_columns(pl.col("read").str.len_chars().alias("read_len"))
    df = df.filter(pl.col("read_len").is_in(list(read_lens)))
    df = df.sort("tr_ID")
    id_lib = df["tr_ID"].unique(maintain_order=True)
    mask_f = f_ids.is_in(id_lib)

    print("Constructing empty datasets...")
    sparse_array = [
        sparse.csr_matrix((num_read_lens, w), dtype=np.int32)
        for w in tqdm(f_lens.filter(~mask_f))
    ]
    riboseq_data = np.empty(len(mask_f), dtype=object)
    riboseq_data[~mask_f] = sparse_array

    tr_mask = id_lib.is_in(f_ids)
    assert tr_mask.all(), (
        "Transcript IDs exist within mapped reads which"
        " are not present in the h5 database. Please apply an identical assembly"
        " for both setting up this database and mapping the ribosome reads."
    )
    print("Aggregating reads...")
    arg_sort = f_ids.arg_sort()
    h5_idxs = arg_sort[f_ids[arg_sort].search_sorted(id_lib)]
    for idx, (_, group) in tqdm(
        zip(h5_idxs, df.group_by("tr_ID", maintain_order=True)),
        total=len(id_lib),
    ):
        tr_reads = np.zeros((num_read_lens, f_lens[idx]), dtype=np.int32)
        for row in group.rows():
            tr_reads[read_len_dict[row[3]], row[1] - 1] += 1
        riboseq_data[idx] = sparse.csr_matrix(tr_reads, dtype=np.int32)

    return riboseq_data
