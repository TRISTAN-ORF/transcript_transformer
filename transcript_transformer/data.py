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
from .util_functions import (
    vec2DNA,
    construct_prot,
    slice_gen,
    prot2vec,
    check_genomic_order,
)
from transcript_transformer import REQ_HEADERS, CUSTOM_HEADERS, DROPPED_HEADERS


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
        print("--> Loading in assembly data...")
        DNA_seq = pyfaidx.Fasta(fa_path)
        gtf = read_gtf(gtf_path, result_type="polars")
        # use biobear instead (does not work well)
        # session = bb.connect()
        # gtf = session.sql(f"SELECT * FROM gtf_scan('{gtf_path}')").to_polars()
        # import exon number as int (strings have wrong sortin (e.g. 10, 11, 2,...))
        gtf = gtf.with_columns(pl.col("exon_number").cast(pl.Int32, strict=False))
        db_tr = parse_transcriptome(gtf, DNA_seq)
        db_gtf = parse_genome(gtf)
        no_handle = True
        max_wait = 900
        waited = 0
        while no_handle and (waited < max_wait):
            try:
                f = h5py.File(h5_path, "a")
                no_handle = False
                try:
                    f = save_transcriptome_to_h5(f, db_tr)
                    if len(db_gtf) > 0:
                        f = save_genome_to_h5(f, db_gtf)
                    f.close()
                    if backup and (not pulled):
                        shutil.copy(h5_path, backup_path)
                except Exception as e:
                    logging.error(traceback.format_exc())
                    print("Failed to update h5 database, which might be corrupted")
            except Exception as e:
                if waited < max_wait:
                    time.sleep(120)
                    waited += 120
        if no_handle:
            print("Could not open h5 database, suspending...")


def process_ribo_data(
    h5_path,
    ribo_paths,
    overwrite=False,
    parallel=False,
    low_memory=False,
):
    # TODO implement option to run custom read lens
    read_lims = [20, 41]
    # load from hdf5 file
    with h5py.File(h5_path, "r") as f:
        tr_ids = pl.Series(np.array(f["transcript/transcript_id"]), dtype=pl.Utf8)
        tr_lens = pl.from_numpy(np.array(f["transcript/transcript_len"])).to_series()
        f_keys = f.keys()
    samples = {}
    for group_samples in ribo_paths.values():
        samples.update(group_samples)
    samples_to_process = deepcopy(samples)
    for sample_id, path in samples.items():
        cond_1 = (
            parallel
            and (not overwrite)
            and (os.path.isfile(h5_path.split(".h5")[0] + f"_{sample_id}.h5"))
        )
        cond_2 = not (parallel or overwrite) and (
            f"transcript/riboseq/{sample_id}" in f_keys
        )
        if cond_1 or cond_2:
            print(
                f"--> {sample_id} in h5, omitting..."
                "(use --overwrite for overwriting existing riboseq data)"
            )
            samples_to_process.pop(sample_id)
    for sample_id, path in samples_to_process.items():
        print(f"--> Loading in {sample_id}...")
        riboseq_data = parse_ribo_reads(path, read_lims, tr_ids, tr_lens, low_memory)
        try:
            print("\t -- Saving data...")
            if not parallel:
                f = h5py.File(h5_path, "a")
            else:
                f = h5py.File(h5_path.split(".h5")[0] + f"_{sample_id}.h5", "w")
                f.create_group("transcript")
            if "riboseq" not in f["transcript"].keys():
                f["transcript"].create_group("riboseq")
            if sample_id in f["transcript/riboseq"].keys():
                del f[f"transcript/riboseq/{sample_id}"]
            f["transcript/riboseq"].create_group(sample_id)
            exp_grp = f[f"transcript/riboseq/{sample_id}"].create_group("5")
            h5max.store_sparse(exp_grp, riboseq_data, format="csr")
            num_reads = [s.sum() for s in riboseq_data]
            exp_grp.create_dataset("num_reads", data=np.array(num_reads).astype(int))
            exp_grp.create_dataset("metadata", data=read_lims)
            f.close()
        except Exception as error:
            print(error)
            del f[f"transcript/riboseq/{sample_id}"]


def save_genome_to_h5(f, db):
    print("--> Saving gene data to hdf5 files...")
    grp = f.create_group("gene")
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
        else:
            grp.create_dataset(key, data=db[key])

    return f


def save_transcriptome_to_h5(f, db):
    print("\t -- Saving data...")
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


def parse_genome(gtf):
    gene_gtf = gtf.filter(pl.col("feature") == "gene")
    cols_to_drop = ["score", "frame"]
    for col in gene_gtf.columns[8:]:
        if gene_gtf.schema[col] == pl.Float64:
            if gene_gtf[col].null_count() == gene_gtf.height or all(
                gene_gtf[col].is_nan()
            ):
                cols_to_drop.append(col)
        else:
            if gene_gtf[col].null_count() == gene_gtf.height:
                cols_to_drop.append(col)

    gene_gtf = gene_gtf.drop(cols_to_drop)

    return gene_gtf


def parse_transcriptome(gtf, DNA_seq):
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

    print("\t -- Reading in assembly...")
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

    altered_tr_exons = []
    print("\t -- Importing transcripts and metadata...")
    for tr_id, gtf_tr in tqdm(
        gtf_set.group_by("transcript_id", maintain_order=True), total=len(db)
    ):
        is_pos_strand = (gtf_tr["strand"] == "+").any()
        # assert start > end
        assert any(
            gtf_tr["start"] <= gtf_tr["end"]
        ), f"Start and end coordinates are not correct for transcript {tr_id}"
        # Check and fix exon ordering
        gtf_tmp = gtf_tr.filter(pl.col("feature") == "exon").sort(
            "start", descending=[not is_pos_strand]
        )
        gtf_tmp = gtf_tmp.with_columns(
            exon_number_alt=pl.Series(np.arange(1, gtf_tmp.height + 1))
        )
        if any(gtf_tmp["exon_number"] != gtf_tmp["exon_number_alt"]):
            exon_dict = dict(
                gtf_tmp.select(["exon_number", "exon_number_alt"]).iter_rows()
            )
            gtf_tr = gtf_tr.with_columns(pl.col("exon_number").replace(exon_dict)).sort(
                "exon_number"
            )
            altered_tr_exons.append(tr_id)
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
                DNA_seq[exon["seqname"]],
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
            CDS_coords = (
                ftrs["CDS"][:, ["start", "end"]]
                .transpose()
                .unpivot()["value"]
                .to_numpy()
            )
            check_genomic_order(CDS_coords, "+" if is_pos_strand else "-")
            data_dict["CDS_coords"].append(CDS_coords)
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
        check_genomic_order(exon_coords, "+" if is_pos_strand else "-")
        data_dict["exon_coords"].append(np.array(exon_coords))
        data_dict["seq"].append(seq)
        data_dict["tis"].append(target_seq)
        data_dict["transcript_id"].append(gtf_tr["transcript_id"].unique()[0])

    if len(altered_tr_exons) > 0:
        print(
            f"WARNING: Exon numbering for {len(altered_tr_exons)} transcripts was altered. "
            "Please check the GTF file for correct exon numbering."
        )
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


def aggregate_sam_file(path, read_lims, low_memory=False):
    schema = {"column_3": pl.Utf8, "column_4": pl.Int32, "column_10": pl.Utf8}
    columns = ["column_3", "column_4", "column_10"]
    new_columns = ["transcript_id", "pos", "read"]
    # Scan complete file in memory
    lf = (
        pl.scan_csv(
            path,
            has_header=False,
            comment_prefix="@",
            schema_overrides=schema,
            low_memory=low_memory,
            separator="\t",
        )
        .select(columns)
        .rename({o: n for o, n in zip(columns, new_columns)})
        # Cast position early if possible (assuming it's non-negative)
        .with_columns(pl.col("pos").cast(pl.UInt32))
    )
    lf_agg = aggregate_reads(lf, read_lims)
    return lf_agg


def aggregate_bam_file(path, read_lims):
    # Use biobear to load data
    s = f"CREATE EXTERNAL TABLE test STORED AS BAM LOCATION '{path}'"
    ctx = bb.connect()
    ctx.sql(s)
    s_2 = f"""
    SELECT reference, start, sequence
    FROM test
    """
    lf = (
        ctx.sql(s_2)
        .to_polars(lazy=True)
        .rename({"reference": "transcript_id", "start": "pos", "sequence": "read"})
        .select(["transcript_id", "pos", "read"])
        .with_columns(pl.col("pos").cast(pl.UInt32))
    )
    lf_agg = aggregate_reads(lf, read_lims)
    ctx.sql("DROP TABLE test")

    return lf_agg


def aggregate_reads(lf, read_lims):
    print("\t -- Filtering on read lens...")
    lf = lf.with_columns(pl.col("read").str.len_chars().alias("read_len"))
    lf = lf.filter(
        (pl.col("read_len") >= read_lims[0]) & (pl.col("read_len") < read_lims[1])
    )
    print("\t -- Aggregating reads...")
    lf_agg = lf.group_by(["transcript_id", "read_len", "pos"]).agg(
        pl.col("read").count().alias("read_count").cast(pl.UInt32)
    )

    return lf_agg


def parse_ribo_reads(path, read_lims, f_ids, f_lens, low_memory=False):
    print(f"\t -- Reading and processing file: {path}...")
    _, file_ext = os.path.splitext(path)

    f_ids_series = pl.Series(f_ids, dtype=pl.Utf8)
    f_lens_series = pl.Series(f_lens, dtype=pl.UInt32)

    if file_ext == ".sam":
        lf_agg = aggregate_sam_file(path, read_lims, low_memory)
    elif file_ext == ".bam":
        lf_agg = aggregate_bam_file(path, read_lims)
    else:
        raise TypeError(f"file extension {file_ext} not supported")

    num_read_lens = read_lims[1] - read_lims[0]
    tr_len_dict = {
        i: l for i, l in zip(f_ids_series.to_list(), f_lens_series.to_list())
    }
    read_len_dict = {
        read_len: i for i, read_len in enumerate(range(read_lims[0], read_lims[1]))
    }

    print("\t -- Lazily filtering for relevant transcript IDs...")
    lf_agg_filtered = lf_agg.filter(pl.col("transcript_id").is_in(f_ids_series))

    # --- Prepare data structures ---
    riboseq_data = {}
    print(f"\t\t -- Pre-constructing {len(f_ids_series)} empty datasets...")
    for tr_id, length in tqdm(
        zip(f_ids_series.to_list(), f_lens_series.to_list()),
        total=len(f_ids_series),
        desc="Initializing matrices",
    ):
        riboseq_data[tr_id] = sparse.csr_matrix((num_read_lens, length), dtype=np.int32)

    # --- Collect the results of the first aggregation (Memory Bottleneck Point) ---
    print("\t -- Collecting filtered aggregated data (streaming)...")
    df_agg = lf_agg_filtered.collect(streaming=True)
    print(f"\t\t -- Collected filtered aggregated data: {df_agg.shape[0]} rows.")
    print(
        f"\t\t -- Collected DataFrame memory usage: {df_agg.estimated_size('mb'):.2f} MB"
    )
    del lf_agg_filtered

    # --- Sort the collected data ---
    if df_agg.height > 0:
        print("\t -- Sorting aggregated data by transcript_id...")
        df_agg_sorted = df_agg.sort("transcript_id")
        del df_agg  # Release memory of the unsorted frame if sort was successful

        print(f"\t -- Iterating over sorted groups...")
        # Group by the sorted column; maintain_order=True improves performance after sort
        grouped = df_agg_sorted.group_by("transcript_id", maintain_order=True)
        num_groups = df_agg_sorted.n_unique("transcript_id")

        for names, group_df in tqdm(
            grouped, total=num_groups, desc="Processing transcripts"
        ):
            transcript_id = names[0]
            # Get the expected length for this transcript
            mtx_width = tr_len_dict.get(transcript_id)
            mtx_shape = (num_read_lens, mtx_width)

            # Extract relevant columns to NumPy arrays for processing
            read_lens_np = group_df["read_len"].to_numpy()
            pos_np = group_df["pos"].to_numpy()  # Assuming this is 1-based from SAM/BAM
            num_reads_np = group_df["read_count"].to_numpy()

            # --- Position Adjustment and Filtering ---
            pos_np_0based = pos_np - 1  # Adjust if input `pos` is already 0-based
            valid_pos_mask = (pos_np_0based >= 0) & (pos_np_0based < mtx_width)

            if not valid_pos_mask.all():
                read_lens_np = read_lens_np[valid_pos_mask]
                pos_np_0based = pos_np_0based[valid_pos_mask]
                num_reads_np = num_reads_np[valid_pos_mask]

            # Create sparse matrix only if there's valid data remaining
            if len(read_lens_np) > 0:
                sp_mtx = create_sparse_matrix_from_arrays(
                    read_lens_np, pos_np_0based, num_reads_np, mtx_shape, read_len_dict
                )
                riboseq_data[transcript_id] = sp_mtx

        del df_agg_sorted

    else:
        print(
            "\t -- No relevant aggregated data found after filtering. Only empty matrices created."
        )

    # --- Finalize output array ---
    print("\t -- Finalizing output array...")
    final_data_list = [riboseq_data[id_key] for id_key in f_ids_series.to_list()]
    return np.array(final_data_list, dtype=object)


def create_sparse_matrix_from_arrays(
    read_lens_np, pos_np_0based, num_reads_np, mtx_shape, read_len_dict
):
    """
    Creates a sparse matrix directly from NumPy arrays.
    Assumes pos_np_0based is already 0-based.
    """
    if len(read_lens_np) == 0:
        return sparse.csr_matrix(mtx_shape, dtype=np.int32)
    try:
        rows = [read_len_dict[read_len] for read_len in read_lens_np]
    except KeyError as e:
        raise ValueError(
            f"Invalid read length {e} found in data, not in read_len_dict mapping. Check read_lims."
        ) from e
    cols = pos_np_0based
    data = num_reads_np
    sparse_matrix_coo = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=mtx_shape,
        dtype=np.int32,
    )
    # Convert to CSR and sum duplicates
    sparse_matrix_csr = sparse_matrix_coo.tocsr()
    return sparse_matrix_csr
