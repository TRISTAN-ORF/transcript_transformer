import os
import shutil
import numpy as np

from scipy import sparse
from tqdm import tqdm
import polars as pl
import pandas as pd

import h5py
import h5max
import pyfaidx
from gtfparse import read_gtf

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
        sl = list(map(lambda x: seq_dict[x], sl))

    if strand in ["-", -1, False]:
        if comp_dict is not None:
            sl = list(map(lambda x: comp_dict[x], sl))[::-1]
        else:
            sl = sl[::-1]

    return np.array(sl)


def process_data(args):
    if not args.backup_path:
        args.backup_path = os.path.splitext(args.gtf_path)[0] + ".h5"
    pulled = False
    if not os.path.isfile(args.h5_path) and os.path.isfile(args.backup_path):
        print(f"Processed assembly data restored ({args.backup_path})")
        shutil.copy(args.backup_path, args.h5_path)
        pulled = True
     
    f = h5py.File(args.h5_path, "a")
    if "transcript" in f.keys():
        print(
            "--> parsed transcriptome directory found, "
            "assembly information can not be re-processed (for existing h5 files)."
        )
    else:
        f.create_group("transcript")
        f = parse_transcriptome(f, args.gtf_path, args.fa_path)
        if args.backup and not pulled:
            f.close()
            shutil.copy(args.h5_path, args.backup_path)
            f = h5py.File(args.h5_path, "a")
    if args.ribo_paths:
        f = parse_ribo_experiments(
            f,
            args.ribo_paths,
            args.overwrite,
            args.low_memory,
        )
    f.close()

# add canonical TIS, canonical TTS
def parse_transcriptome(f, gtf_path, fa_path):
    print("Loading assembly data...")
    genome = pyfaidx.Fasta(fa_path)
    contig_list = pd.Series(genome.keys())
    gtf = read_gtf(gtf_path)
    gtf = gtf.with_columns(pl.col("exon_number").cast(pl.Int32, strict=False))
    headers = {"transcript_id":"id", "gene_id": "gene_id", "gene_name": "gene_name", 
               "strand": "strand", "transcript_biotype": "biotype", "tag": "tag", 
               "transcript_support_level": "support_lvl"}
    
    print("Extracting transcripts and metadata...")
    data_dict = {"id": [], "seq": [], "tis": [], "gene_id": [], "gene_name": [], "contig": [],
                    "strand": [], "biotype": [], "tag": [], "support_lvl": [], "canonical_TIS_exon_idx": [],
                    "exon_idxs": [], "exon_coords": [], "canonical_TIS_idx": [], "canonical_TIS_coord": [], 
                    "canonical_TTS_idx": [], "canonical_TTS_coord": [], "tr_len": [], "canonical_prot_id": []}
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
                gtf_tr.filter(pl.col("feature") == "transcript")
                .select(list(headers.keys()))).melt().to_dict().values()
            for k, i in zip(list(headers.values()), values):
                data_dict[k].append(i)
                
            # obtain and sort exon information (strings have wrong sortin (e.g. 10, 11, 2, 3, ...))
            exons = gtf_tr.filter(pl.col("feature") == "exon")
            exon_lens = (abs(exons["start"] - exons["end"]) + 1).to_numpy()
            strand_is_pos = (exons["strand"] == "+").any()
            cum_exon_lens = np.insert(np.cumsum(exon_lens), 0, 0)
            data_dict['tr_len'].append(exon_lens.sum())

            if len(exons) == 0:
                print(
                    "WARNING: No exons found for transcript. This should not happen. Please ensure"
                    "exons are marked with the correct transcript id"
                )

            # obtain TISs, select first in case of split (intron) start codon
            # TODO: when multiple TISs are supported, alter code
            start_codon = (
                gtf_tr.filter(pl.col("feature") == "start_codon").slice(0, 1).to_dicts()
            )
            stop_codon = (
                gtf_tr.filter(pl.col("feature") == "stop_codon").slice(0, 1).to_dicts()
            )

            CDSs = gtf_tr.filter(pl.col("feature") == "CDS")
            cds_length = abs(CDSs["start"].sum() - CDSs["end"].sum()) + len(CDSs)
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
                data_dict["canonical_TTS_idx"].append(tis_idx + cds_length)
                prot_id = gtf_tr["protein_id"].unique(maintain_order=True)[1]
                data_dict["canonical_prot_id"].append(prot_id)
                data_dict["canonical_TIS_coord"].append(tis)
                data_dict["canonical_TTS_coord"].append(tts)
            else:
                data_dict["canonical_TIS_exon_idx"].append(-1)
                data_dict["canonical_TIS_idx"].append(-1)
                data_dict["canonical_TTS_idx"].append(-1)
                data_dict["canonical_prot_id"].append("")
                data_dict["canonical_TIS_coord"].append(-1)
                data_dict["canonical_TTS_coord"].append(-1)

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
            data_dict['exon_idxs'].append(exon_idxs)
            data_dict["exon_coords"].append(np.array(exon_coords))
            data_dict['seq'].append(np.concatenate(exon_seqs))
            data_dict['tis'].append(target_seq)
            data_dict['id'].append(id)
            data_dict['contig'].append(contig)

    print("Save data in hdf5 files...")
    dt8 = h5py.vlen_dtype(np.dtype("int8"))
    dt = h5py.vlen_dtype(np.dtype("int"))
    grp = f["transcript"]
    for key, array in data_dict.items():
        if key in ["id", "contig", "gene_id", "gene_name", "strand", 
                   "biotype", "tag", "support_lvl", "canonical_prot_id"]:
            grp.create_dataset(key, data=array, dtype=f"<S{max([len(s) for s in array])}")
        elif key in ["seq", "tis"]:
            grp.create_dataset(key, data=array, dtype=dt8)
        elif key in ["exon_idxs", "exon_coords"]:
            grp.create_dataset(key, data=np.array(array, dtype=object), dtype=dt)
        else:
            grp.create_dataset(key, data=array)

    return f


def parse_ribo_experiments(f, ribo_paths, overwrite=False, low_memory=False):
    if "riboseq" not in f["transcript"].keys():
        f["transcript"].create_group("riboseq")

    tr_ids = np.array(f["transcript/id"])
    tr_lens = np.array(f["transcript/tr_len"])
    header_dict = {2: "tr_ID", 3: "pos", 9: "read"}

    for experiment, path in ribo_paths.items():
        if experiment in f["transcript/riboseq"].keys():
            if overwrite:
                del f[f"transcript/riboseq/{experiment}"]
            else:
                print(
                    f"--> {experiment} in h5, omitting..."
                    "(use --overwrite for overwriting existing riboseq data)"
                )
                continue
        try:
            print(f"Loading in {experiment}...")
            df = pl.read_csv(
                path,
                has_header=False,
                comment_char="@",
                columns=[2, 3, 9],
                sep="\t",
                low_memory=low_memory,
            )
            df.columns = list(header_dict.values())
            f["transcript/riboseq"].create_group(experiment)
            exp_grp = f[f"transcript/riboseq/{experiment}"].create_group("5")
            # TODO implement option to run custom read lens
            read_lens = np.arange(20, 41)
            riboseq_data = parse_ribo_reads(df, read_lens, tr_ids, tr_lens)
            print("Saving data...")
            h5max.store_sparse(exp_grp, riboseq_data, format="csr")
            num_reads = [s.sum() for s in riboseq_data]
            exp_grp.create_dataset("num_reads", data=np.array(num_reads).astype(int))
            exp_grp.create_dataset("metadata", data=read_lens)

            print("Data processing completed.")
        except Exception as error:
            print(error)
            del f[f"transcript/riboseq/{experiment}"]

    return f


def parse_ribo_reads(df, read_lens, tr_ids, tr_lens):
    num_read_lens = len(read_lens)
    read_len_dict = {read_len: i for i, read_len in enumerate(read_lens)}
    print("Filtering on read lens...")
    df = df.with_columns(pl.col("read").str.lengths().alias("read_len"))
    df = df.filter(pl.col("read_len").is_in(list(read_lens)))
    ID_unique = df["tr_ID"].unique()
    mask_f = np.isin(tr_ids, ID_unique.to_numpy().astype("S"))

    print("Constructing empty datasets...")
    sparse_array = [
        sparse.csr_matrix((num_read_lens, w)) for w in tqdm(tr_lens[~mask_f])
    ]
    riboseq_data = np.empty(len(mask_f), dtype=object)
    riboseq_data[~mask_f] = sparse_array
    df = df.sort("tr_ID")

    print("Aggregating reads...")
    for tr_id, group in tqdm(df.groupby("tr_ID"), total=len(ID_unique)):
        mask_tr = tr_ids == tr_id.encode()
        tr_reads = np.zeros((num_read_lens, tr_lens[mask_tr][0]), dtype=np.uint32)
        for row in group.rows():
            tr_reads[read_len_dict[row[3]], row[1] - 1] += 1
        riboseq_data[mask_tr] = sparse.csr_matrix(tr_reads)

    return riboseq_data


if __name__ == "__main__":
    f = h5py.File("../test/test.h5", "w")
    f.create_group("transcript")
    parse_transcriptome(
        f,
        "../test/data/GRCh38v107_snippet.gtf",
        "../test/data/GRCh38_snippet.fa",
    )
    f.close()
