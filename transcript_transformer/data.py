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
import pyarrow as pa

import h5py
import h5max
import pyfaidx
from gtfparse import read_gtf
from .util_functions import vec2DNA, construct_prot, time, slice_gen, prot2vec
from pdb import set_trace

REQ_HEADERS = [
    "seqname",
    "feature",
    "start",
    "end",
    "strand",
    "gene_id",
    "transcript_id",
    "exon_number",
]
CUSTOM_HEADERS = [
    "transcript_id",
    "seq",
    "tis",
    "canonical_TIS_exon",
    "exon_idxs",
    "exon_coords",
    "CDS_idxs",
    "CDS_coords",
    "has_annotated_start_codon",
    "has_annotated_stop_codon",
    "canonical_TIS_idx",
    "canonical_TIS_coord",
    "canonical_TTS_idx",
    "canonical_TTS_coord",
    "canonical_LTS_idx",
    "canonical_LTS_coord",
    "transcript_len",
    "canonical_protein_seq",
]
DROPPED_HEADERS = [
    "end",
    "exon_id",
    "exon_version",
    "exon_number",
    "feature",
    "frame",
    "score",
    "start",
]


def process_seq_data(h5_path, gtf_path, fa_path, backup_path, backup=True):
    pulled = False
    if not backup_path:
        backup_path = os.path.splitext(gtf_path)[0] + ".h5"
    if os.path.abspath(backup_path) == os.path.abspath(h5_path):
        print(
            f"!-> Backup path identical to h5 output path, no database copy will be created..."
        )
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
        db = parse_transcriptome(gtf_path, fa_path)
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
                f = save_transcriptome_to_h5(f, db)
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
    tr_ids = (
        pl.from_numpy(np.array(f["transcript/transcript_id"])).to_series().cast(pl.Utf8)
    )
    tr_lens = pl.from_numpy(np.array(f["transcript/transcript_len"])).to_series()
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
            schema = {"column_3": pl.Utf8, "column_4": pl.Int32, "column_10": pl.Utf8}
            lf = pl.scan_csv(
                path,
                has_header=False,
                comment_prefix="@",
                schema_overrides=schema,
                separator="\t",
            ).select(["column_3", "column_4", "column_10"])
        elif file_ext == ".bam":
            s = f"CREATE EXTERNAL TABLE test STORED AS BAM LOCATION '{path}'"
            ctx = bb.connect()
            ctx.sql(s)
            exe = ctx.sql("SELECT reference, start, sequence FROM test")
            # Convert the list of RecordBatches to a Table
            table = pa.Table.from_batches(exe.to_arrow_record_batch_reader())
            # Create a Dataset from the Table
            dataset = pa.dataset.dataset(table)
            # Lazyframe
            lf = pl.scan_pyarrow_dataset(dataset)

        else:
            raise TypeError(f"file extension {file_ext} not supported")
        new_columns = ["transcript_id", "pos", "read"]
        lf = lf.rename({o: n for o, n in zip(lf.columns, new_columns)})

        # TODO implement option to run custom read lens
        read_lens = np.arange(20, 41)
        riboseq_data = parse_ribo_reads(lf, read_lens, tr_ids, tr_lens)
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


def save_transcriptome_to_h5(f, db):
    print("Save data in hdf5 files...")
    dt8 = h5py.vlen_dtype(np.dtype("int8"))
    dt = h5py.vlen_dtype(np.dtype("int"))
    grp = f.create_group("transcript")
    for key in db.columns:
        if db[key].dtype == pl.Categorical:
            db = db.with_columns(pl.col(key).cast(pl.String))
        if db[key].dtype == pl.String:
            array = [a if a != None else "" for a in db[key]]
            max_char_len = db[key].str.len_chars().max()
            if max_char_len > 0:
                grp.create_dataset(key, data=array, dtype=f"<S{max_char_len}")
            else:
                continue
        elif key in ["seq", "tis", "canonical_protein_seq"]:
            grp.create_dataset(key, data=db[key], dtype=dt8)
        elif key in ["exon_idxs", "exon_coords", "CDS_idxs", "CDS_coords"]:
            grp.create_dataset(key, data=np.array(db[key], dtype=object), dtype=dt)
        else:
            grp.create_dataset(key, data=db[key])

    return f


def parse_transcriptome(gtf_path, fa_path):
    print("--> Loading assembly data...")
    genome = pyfaidx.Fasta(fa_path)
    contig_list = pl.Series(genome.keys())
    gtf = read_gtf(gtf_path, result_type="polars")
    # use biobear instead
    # session = bb.connect()
    # gtf = session.sql(f"SELECT * FROM gtf_scan('{gtf_path}')").to_polars()

    # import exon number as int (strings have wrong sortin (e.g. 10, 11, 2,...))
    gtf = gtf.with_columns(pl.col("exon_number").cast(pl.Int32, strict=False))
    # ensure all required fields are listed
    assert np.isin(
        REQ_HEADERS, gtf.columns
    ).all(), f"Not all required properties in gtf file: {REQ_HEADERS}"
    # evaluate extra columns
    xtr_cols = np.array(gtf.columns)[
        ~pl.Series(gtf.columns).is_in(REQ_HEADERS).to_numpy()
    ]
    data_dict_keys = np.array(REQ_HEADERS + CUSTOM_HEADERS + list(xtr_cols))
    data_dict = {k: [] for k in CUSTOM_HEADERS}
    data_cols_in_gtf = data_dict_keys[np.isin(data_dict_keys, gtf.columns)]

    print("--> Importing transcripts and metadata...")
    gtf_set = gtf.filter(
        # exclude transcript ids that are empty
        pl.col("transcript_id") != "",
        pl.col("feature").is_in(
            ["transcript", "exon", "CDS", "start_codon", "stop_codon"]
        ),
    ).sort(["seqname", "transcript_id", "exon_number"])
    gtf_set = gtf_set.with_columns(
        (abs(pl.col("start") - pl.col("end")) + 1).alias("feature_length")
    )
    trs = gtf_set["transcript_id"].unique(maintain_order=True)

    db = pl.DataFrame(data={"transcript_id": trs})
    db = db.join(gtf.filter(pl.col("feature") == "transcript"), on="transcript_id")

    for tr_id, gtf_tr in tqdm(
        gtf_set.group_by("transcript_id", maintain_order=True), total=len(db)
    ):
        is_pos_strand = (gtf_tr["strand"] == "+").any()
        ftrs = {}
        ftr_cum_lens = {}
        ftr_idxs = {}
        for feature, feature_df in gtf_tr.group_by("feature", maintain_order=True):
            ftrs[feature[0]] = feature_df
            ftr_lens = feature_df["feature_length"].drop_nulls().to_numpy()
            cum_lens = np.insert(np.cumsum(ftr_lens), 0, 0)
            ftr_cum_lens[feature[0]] = cum_lens
            # feature boundaries; tuples flattened into single vector (e.g. [0,10,10,12,12,20])
            ftr_idxs[feature[0]] = np.vstack((cum_lens[:-1], cum_lens[1:])).T.ravel()

        data_dict["transcript_len"].append(ftr_cum_lens["exon"].max())
        if ftr_cum_lens["exon"].max() == 0:
            print(
                "WARNING: No exons found for transcript. This should not happen. Please ensure"
                "exons are marked with the correct transcript id"
            )
        # TODO: when multiple TISs are supported, code needs update
        # init empty boolean to denote TIS locations
        target_seq = np.full(ftr_cum_lens["exon"].max(), False)

        exon_coords = []
        exon_seqs = []
        for exon_i, exon in enumerate(ftrs["exon"].iter_rows(named=True)):
            # get sequence
            exon_seq = slice_gen(
                genome[exon["seqname"]],
                exon["start"],
                exon["end"],
                exon["strand"],
                to_vec=True,
            ).astype(np.int16)
            exon_coords.append(exon["start"])
            exon_coords.append(exon["end"])
            exon_seqs.append(exon_seq)
        seq = np.concatenate(exon_seqs)

        if "CDS" in ftrs:
            # select first in case of split (intron) start codon
            first_cds = ftrs["CDS"][0].to_dicts()[0]
            exon_i = first_cds["exon_number"] - 1
            exon = ftrs["exon"][exon_i].to_dicts()[0]
            # shift CDS transcript idxs based on start exon
            exon_shift = ftr_cum_lens["exon"][exon_i]
            if is_pos_strand:
                # shift CDS transcript idxs based on cds start in exon
                in_exon_shift = ftrs["CDS"][0, "start"] - exon["start"]
                tis = first_cds["start"]
                tis_idx = ftr_cum_lens["exon"][exon_i] + tis - exon["start"]
                lts = ftrs["CDS"][-1].to_dicts()[0]["end"]
                if "stop_codon" in ftrs:
                    tts = ftrs["stop_codon"][0][0, "start"]
                else:
                    tts = -1
            else:
                # shift CDS transcript idxs based on cds start in exon
                in_exon_shift = exon["end"] - ftrs["CDS"][0, "end"]
                tis = first_cds["end"]
                tis_idx = ftr_cum_lens["exon"][exon_i] + exon["end"] - tis
                lts = ftrs["CDS"][-1].to_dicts()[0]["start"]
                if "stop_codon" in ftrs:
                    tts = ftrs["stop_codon"][0][0, "end"]
                else:
                    tts = -1
            target_seq[tis_idx] = 1
            DNA_frag = vec2DNA(seq[tis_idx:])
            prot, _, _ = construct_prot(DNA_frag)
            data_dict["has_annotated_stop_codon"].append("stop_codon" in ftrs)
            data_dict["has_annotated_start_codon"].append("start_codon" in ftrs)
            data_dict["CDS_idxs"].append(ftr_idxs["CDS"] + exon_shift + in_exon_shift)
            data_dict["CDS_coords"].append(
                ftrs["CDS"][:, ["start", "end"]]
                .transpose()
                .unpivot()["value"]
                .to_numpy()
            )
            data_dict["canonical_TIS_exon"].append(exon_i + 1)
            data_dict["canonical_TIS_idx"].append(tis_idx)
            # LTS: Last Translation Site; 1 nucleotide upstream of TTS
            tts_idx = tis_idx + ftr_cum_lens["CDS"].max()
            data_dict["canonical_TTS_idx"].append(tts_idx)
            data_dict["canonical_LTS_idx"].append(tts_idx - 1)
            data_dict["canonical_TIS_coord"].append(tis)
            data_dict["canonical_TTS_coord"].append(tts)
            data_dict["canonical_LTS_coord"].append(lts)
            data_dict["canonical_protein_seq"].append(prot)
        else:
            data_dict["has_annotated_stop_codon"].append(False)
            data_dict["has_annotated_start_codon"].append(False)
            data_dict["CDS_idxs"].append(np.empty(0, dtype=int))
            data_dict["CDS_coords"].append(np.empty(0, dtype=int))
            data_dict["canonical_TIS_exon"].append(-1)
            data_dict["canonical_TIS_idx"].append(-1)
            data_dict["canonical_TTS_idx"].append(-1)
            data_dict["canonical_LTS_idx"].append(-1)
            data_dict["canonical_TIS_coord"].append(-1)
            data_dict["canonical_TTS_coord"].append(-1)
            data_dict["canonical_LTS_coord"].append(-1)
            data_dict["canonical_protein_seq"].append(None)
        data_dict["exon_idxs"].append(ftr_idxs["exon"])
        data_dict["exon_coords"].append(np.array(exon_coords))
        data_dict["seq"].append(seq)
        data_dict["tis"].append(target_seq)
        data_dict["transcript_id"].append(gtf_tr["transcript_id"].unique()[0])

    db_ext = pl.from_dict(data_dict)
    db = db_ext.join(db, on="transcript_id", how="left")
    # drop exon info that is not correct at transcript-level
    db = db.drop(DROPPED_HEADERS, strict=False)
    # vectorize protein sequences (less storage)
    db = db.with_columns(
        pl.col("canonical_protein_seq")
        .fill_null("")
        .map_elements(
            prot2vec,
            pl.List(pl.Int64),
        )
        .cast(pl.List(pl.Int8))
    )

    return db


def parse_ribo_reads(lf, read_lens, f_ids, f_lens):
    num_read_lens = len(read_lens)
    tr_len_dict = {i: l for i, l in zip(f_ids, f_lens)}
    read_len_dict = {read_len: i for i, read_len in enumerate(read_lens)}
    print("Filtering on read lens...")
    lf = lf.with_columns(pl.col("read").str.len_chars().alias("read_len"))
    lf = lf.filter(pl.col("read_len").is_in(list(read_lens)))
    id_lib = lf.select("transcript_id").unique().collect()
    mask_f = f_ids.is_in(id_lib)

    print("Constructing empty datasets...")
    riboseq_data = {
        tr_id: sparse.csr_matrix((num_read_lens, w), dtype=np.int32)
        for tr_id, w in zip(f_ids.filter(~mask_f), f_lens.filter(~mask_f))
    }
    tr_mask = id_lib["transcript_id"].is_in(f_ids)
    assert tr_mask.all(), (
        "Transcript IDs exist within mapped reads which"
        " are not present in the h5 database. Please apply an identical assembly"
        " for both setting up this database and mapping the ribosome reads."
    )
    print("Aggregating reads...")
    lf = lf.group_by("transcript_id", "read_len", "pos").agg(pl.col("read").len())
    lf = lf.group_by("transcript_id").agg(
        pl.col("read_len"), pl.col("pos"), pl.col("read")
    )

    lf = lf.collect()
    for row in tqdm(lf.iter_rows(), total=len(lf)):
        tr_reads = np.zeros((num_read_lens, tr_len_dict[row[0]]), dtype=np.int32)
        for read_len, pos, num_reads in zip(row[1], row[2], row[3]):
            tr_reads[read_len_dict[read_len], pos - 1] = num_reads
        riboseq_data[row[0]] = sparse.csr_matrix(tr_reads, dtype=np.int32)

    return np.array([riboseq_data[id] for id in f_ids])
