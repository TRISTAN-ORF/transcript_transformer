import os
import sys
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
        start, end = co_to_idx(start, end, strand)
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
    f = h5py.File(f"{args.h5_path}", "a")
    if "transcript" in f.keys():
        print(
            "--> parsed transcriptome directory found, "
            "assembly information can not be re-processed (for existing h5 files)."
        )
    else:
        f.create_group("transcript")
        f = parse_transcriptome(f, args.gtf_path, args.fa_path, args.meta_dir)
    if args.ribo_paths:
        f = parse_ribo_experiments(
            f,
            args.ribo_paths,
            args.overwrite,
            args.low_memory,
        )

    f.close()


def parse_transcriptome(f, gtf_path, fa_path, meta_dir=None):
    print("Loading assembly data...")
    genome = pyfaidx.Fasta(fa_path)
    contig_list = pd.Series(genome.keys())
    gtf = read_gtf(gtf_path)
    gtf = gtf.with_columns(pl.col("exon_number").cast(pl.Int32, strict=False))
    if meta_dir is None:
        meta_dir = os.path.join(os.path.dirname(gtf_path), "transcript_transformer")
        os.makedirs(meta_dir, exist_ok=True)

    print("Extracting transcripts and metadata...")
    for contig in contig_list:
        print(f"{contig}...")
        if os.path.isfile(os.path.join(meta_dir, f"{contig}.npy")):
            print(
                f"--> Parsed chromosome {contig} in metadata folder ({meta_dir}/), omitting..."
            )
            continue
        tran_ids = []
        samples = []
        poi_contig = []
        gtf_set = gtf.filter(pl.col("seqname") == str(contig))
        tr_set = gtf_set["transcript_id"].unique()
        tr_set = tr_set.filter(tr_set != "")
        tr_datas = []
        headers = [
            "transcript_id",
            "gene_id",
            "gene_name",
            "strand",
            "transcript_biotype",
            "tag",
            "transcript_support_level",
            "exon_number",
        ]

        for i, tr_idx in tqdm(enumerate(tr_set), total=len(tr_set)):
            # obtain transcript information
            gtf_tr = gtf_set.filter(pl.col("transcript_id") == tr_idx).sort(
                by="exon_number"
            )
            tr_data = (
                gtf_tr.filter(pl.col("feature") == "transcript")
                .select(headers)
                .to_pandas()
                .squeeze()
            )

            # obtain and sort exon information (strings have wrong sortin (e.g. 10, 11, 2, 3, ...))
            exons = gtf_tr.filter(pl.col("feature") == "exon")
            exon_lens = (abs(exons["start"] - exons["end"]) + 1).to_numpy()
            strand_is_pos = (exons["strand"] == "+").any()
            cum_exon_lens = np.insert(np.cumsum(exon_lens), 0, 0)
            tr_len = exon_lens.sum()

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

            exon_seqs = []
            poi_tr = []
            target_seq = np.full(tr_len, False)

            if len(start_codon) > 0:
                # use as index for sorted dfs
                start_codon = start_codon[0]
                exon_i = start_codon["exon_number"] - 1
                exon = exons[exon_i].to_dicts()[0]
                if strand_is_pos:
                    tis = start_codon["start"]
                    tis_idx = cum_exon_lens[exon_i] + tis - exon["start"]
                else:
                    tis = start_codon["end"] - 1
                    tis_idx = cum_exon_lens[exon_i] + exon["end"] - tis
                tis_coords = [tis, tis + 1]
                target_seq[tis_idx] = 1
                tr_data["tis_idx"] = tis_idx
                tr_data["tts_idx"] = tis_idx + cds_length
                tr_data["prot_ID"] = gtf_tr["protein_id"].unique(maintain_order=True)[1]
                poi_tr.append(["tis"] + [tis_idx, tis_idx + 1] + tis_coords)

                if len(stop_codon) > 0:
                    stop_codon = stop_codon[0]
                    tts_coords = [stop_codon["end"] - 1, stop_codon["end"]]
                else:
                    exon = exons[-1].to_dicts()[0]
                    if strand_is_pos:
                        tts = exon["start"]
                    else:
                        tts = exon["end"] - 1
                    tts_coords = [tts, tts + 1]

                poi_tr.append(
                    ["tts"] + [tr_data["tts_idx"], tr_data["tts_idx"] + 1] + tts_coords
                )

            for exon_i, exon in enumerate(exons.iter_rows(named=True)):
                # get sequence
                exon_seq = slice_gen(
                    genome[contig],
                    exon["start"],
                    exon["end"],
                    exon["strand"],
                    to_vec=True,
                ).astype(np.int16)
                # switch start/end in case of neg strand
                exon_coords = [exon["start"], exon["end"]]
                exon_coords.sort()
                # compile metadata
                poi_tr.append(
                    [
                        "exon",
                        cum_exon_lens[exon_i],
                        cum_exon_lens[exon_i] + len(exon_seq),
                    ]
                    + exon_coords
                )
                exon_seqs.append(exon_seq)

            tr_datas.append(tr_data)
            samples.append(np.vstack((np.concatenate(exon_seqs), target_seq)))
            tran_ids.append(tr_idx)
            poi_contig.append(poi_tr)

        final = np.array([tran_ids, samples, poi_contig], dtype=object).T
        if len(tr_datas) == 0:
            metadata = pd.DataFrame(columns=headers)
        else:
            metadata = pd.concat(tr_datas, axis=1).T
        metadata.to_csv(os.path.join(meta_dir, f"{contig}.csv"))
        np.save(os.path.join(meta_dir, f"{contig}.npy"), final)

    print("Save data in hdf5 files...")
    dt = h5py.vlen_dtype(np.dtype("int8"))
    contig_type = f"<S{contig_list.str.len().max()}"
    if "metadata" not in f["transcript"]:
        f["transcript"].create_group("metadata")

    grp = f["transcript"]
    seqs = []
    tiss = []
    contigs = []
    ids = []
    len_trs = []
    for contig in contig_list:
        x = np.load(os.path.join(meta_dir, f"{contig}.npy"), allow_pickle=True)
        seqs += [t[0] for t in x[:, 1]]
        tiss += [t[1] for t in x[:, 1]]
        contigs.append(np.full(len(x), contig, dtype=contig_type))
        ids.append(x[:, 0].astype("S15"))

    grp.create_dataset("seq", data=seqs, dtype=dt)
    grp.create_dataset("tis", data=tiss, dtype=dt)
    grp.create_dataset("contig", data=np.hstack(contigs))
    grp.create_dataset("id", data=np.hstack(ids))

    len_trs += [len(seq) for seq in seqs]
    grp.create_dataset("tr_len", data=len_trs)

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
        meta_dir=None,
    )
    f.close()
