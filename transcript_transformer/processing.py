import numpy as np
from tqdm import tqdm
import h5py
import h5max
import pandas as pd
import polars as pl


from .util_functions import (
    construct_prot,
    time,
    vec2DNA,
    find_distant_exon_coord,
    transcript_region_to_exons,
)

HEADERS = [
    "id",
    "contig",
    "biotype",
    "strand",
    "canonical_TIS_coord",
    "canonical_TIS_exon_idx",
    "canonical_TIS_idx",
    "canonical_TTS_coord",
    "canonical_TTS_idx",
    "canonical_protein_id",
    "exon_coords",
    "exon_idxs",
    "gene_id",
    "gene_name",
    "support_lvl",
    "tag",
    "tr_len",
]


OUT_HEADERS = [
    "seqname",
    "ORF_id",
    "tr_id",
    "TIS_pos",
    "output",
    "output_rank",
    "seq_output",
    "start_codon",
    "stop_codon",
    "ORF_len",
    "TTS_pos",
    "TTS_on_transcript",
    "reads_in_tr",
    "reads_in_ORF",
    "reads_out_ORF",
    "in_frame_read_perc",
    "ORF_type",
    "ORF_equals_CDS",
    "tr_biotype",
    "tr_support_lvl",
    "tr_tag",
    "tr_len",
    "dist_from_canonical_TIS",
    "frame_wrt_canonical_TIS",
    "correction",
    "TIS_coord",
    "TIS_exon",
    "TTS_coord",
    "TTS_exon",
    "strand",
    "gene_id",
    "gene_name",
    "canonical_TIS_coord",
    "canonical_TIS_exon_idx",
    "canonical_TIS_idx",
    "canonical_TTS_coord",
    "canonical_TTS_idx",
    "canonical_protein_id",
    "protein_seq",
]

DECODE = [
    "seqname",
    "tr_id",
    "tr_biotype",
    "tr_support_lvl",
    "tr_tag",
    "strand",
    "gene_id",
    "gene_name",
    "canonical_protein_id",
]

RIBOTIE_MQC_HEADER = """
# parent_id: 'ribotie'
# parent_name: "RiboTIE"
# parent_description: "Overview of open reading frames called as translating by RiboTIE"
# """

START_CODON_MQC_HEADER = """
# id: 'ribotie_start_codon_counts' 
# section_name: 'Start Codon'
# description: "Start codon counts of all open reading frames called by RiboTIE"
# plot_type: 'bargraph'
# anchor: 'orf_start_codon_counts'
# pconfig:
#     id: "orf_start_codon_counts_plot"
#     title: "RiboTIE: Start Codons"
#     colors:
#          ATG : "#f8d7da"
#     xlab: "# ORFs"
#     cpswitch_counts_label: "Number of ORFs"
"""

BIOTYPE_VARIANT_MQC_HEADER = """
# id: 'ribotie_biotype_counts_variant'
# section_name: 'Transcript Biotypes (CDS Variant)'
# description: "Transcript biotypes of 'CDS variant' (see ORF types) open reading frames called by RiboTIE"
# plot_type: 'bargraph'
# anchor: 'transcript_biotype_variant_counts'
# pconfig:
#     id: "transcript_biotype_counts_variant_plot"
#     title: "RiboTIE: Transcript Biotypes (CDS Variant)"
#     xlab: "# ORFs"
#     cpswitch_counts_label: "Number of ORFs"
"""

ORF_TYPE_MQC_HEADER = """
# id: 'ribotie_orftype_counts'
# section_name: 'ORF types'
# description: "ORF types of all open reading frames called by RiboTIE"
# plot_type: 'bargraph'
# anchor: 'transcript_orftype_counts'
# pconfig:
#     id: "transcript_orftype_counts_plot"
#     title: "RiboTIE: ORF Types"
#     xlab: "# ORFs"
#     cpswitch_counts_label: "Number of ORFs"
"""

ORF_LEN_MQC_HEADER = """
# id: 'ribotie_orflen_hist'
# section_name: 'ORF lengths'
# description: "ORF lengths of all open reading frames called by RiboTIE"
# plot_type: 'linegraph'
# anchor: 'transcript_orflength_hist'
# pconfig:
#     id: "transcript_orflength_hist_plot"
#     title: "RiboTIE: ORF lengths"
#     xlab: "Length"
#     xLog: "True"
"""

ORF_TYPE_ORDER = [
    "annotated CDS",
    "N-terminal truncation",
    "N-terminal extension",
    "CDS variant",
    "uORF",
    "uoORF",
    "dORF",
    "doORF",
    "intORF",
    "lncRNA-ORF",
    "other",
]

ORF_BIOTYPE_ORDER = [
    "retained_intron",
    "protein_coding",
    "protein_coding_CDS_not_defined",
    "nonsense_mediated_decay",
    "processed_pseudogene",
    "unprocessed_pseudogene",
    "transcribed_unprocessed_pseudogene",
    "transcribed_processed_pseudogene",
    "translated_processed_pseudogene",
    "transcribed_unitary_pseudogene",
    "processed_transcript",
    "TEC",
    "artifact",
    "non_stop_decay",
    "misc_RNA",
]


def construct_output_table(
    h5_path,
    out_prefix,
    prob_cutoff=0.15,
    correction=False,
    dist=9,
    start_codons=".*TG$",
    min_ORF_len=15,
    remove_duplicates=True,
    exclude_invalid_TTS=True,
    ribo=None,
    parallel=False,
):
    f = h5py.File(h5_path, "r")["transcript"]
    f_tr_ids = np.array(f["id"])
    has_seq_output = "seq_output" in f.keys()
    has_ribo_output = ribo is not None
    assert has_seq_output or has_ribo_output, "no model predictions found"
    print(f"--> Processing {out_prefix}...")

    if has_ribo_output:
        tr_ids = np.array([o[0].split(b"|")[1] for o in ribo])
        ribo_id = ribo[0][0].split(b"|")[0]
        xsorted = np.argsort(f_tr_ids)
        pred_to_h5_args = xsorted[np.searchsorted(f_tr_ids[xsorted], tr_ids)]
        preds = np.hstack([o[1] for o in ribo])

    else:
        mask = [len(o) > 0 for o in np.array(f["seq_output"])]
        preds = np.hstack(np.array(f["seq_output"]))
        pred_to_h5_args = np.where(mask)[0]
    # map pred ids to database id
    k = (preds > prob_cutoff).sum()
    if k == 0:
        print(
            f"!-> No predictions with an output probability higher than {prob_cutoff}"
        )
        return None
    lens = np.array(f["tr_len"])[pred_to_h5_args]
    cum_lens = np.cumsum(np.insert(lens, 0, 0))
    idxs = np.argpartition(preds, -k)[-k:]

    orf_dict = {"f_idx": [], "TIS_idx": []}
    for idx in idxs:
        idx_tr = np.where(cum_lens > idx)[0][0] - 1
        orf_dict["TIS_idx"].append(idx - cum_lens[idx_tr])
        orf_dict["f_idx"].append(pred_to_h5_args[idx_tr])
    if has_seq_output:
        seq_out = [
            f["seq_output"][i][j]
            for i, j in zip(orf_dict["f_idx"], orf_dict["TIS_idx"])
        ]
        orf_dict.update({"seq_output": seq_out})
    orf_dict.update(
        {f"{header}": np.array(f[f"{header}"])[orf_dict["f_idx"]] for header in HEADERS}
    )
    orf_dict.update(orf_dict)
    df_out = pd.DataFrame(data=orf_dict)
    df_out = df_out.rename(
        columns={
            "id": "tr_id",
            "support_lvl": "tr_support_lvl",
            "biotype": "tr_biotype",
            "tag": "tr_tag",
        }
    )
    df_out["correction"] = np.nan

    df_dict = {
        "start_codon": [],
        "stop_codon": [],
        "prot": [],
        "TTS_exon": [],
        "TTS_on_transcript": [],
        "TIS_coord": [],
        "TIS_exon": [],
        "TTS_coord": [],
        "TTS_coord": [],
        "TTS_pos": [],
        "ORF_len": [],
    }

    TIS_idxs = df_out.TIS_idx.copy()
    corrections = df_out.correction.copy()
    for i, row in tqdm(
        df_out.iterrows(), total=len(df_out), desc=f"{time()}: parsing ORF information "
    ):
        tr_seq = f["seq"][row.f_idx]
        TIS_idx = row.TIS_idx
        if correction and not np.array_equal(tr_seq[TIS_idx : TIS_idx + 3], [0, 1, 3]):
            low_bound = max(0, TIS_idx - (dist * 3))
            tr_seq_win = tr_seq[low_bound : TIS_idx + (dist + 1) * 3]
            atg = [0, 1, 3]
            matches = [
                x
                for x in range(len(tr_seq_win))
                if np.array_equal(tr_seq_win[x : x + len(atg)], atg)
            ]
            matches = np.array(matches) - min(TIS_idx, dist * 3)
            matches = matches[matches % 3 == 0]
            if len(matches) > 0:
                match = matches[np.argmin(abs(matches))]
                corrections[row.name] = match
                TIS_idx = TIS_idx + match
                TIS_idxs[row.name] = TIS_idx
        DNA_frag = vec2DNA(tr_seq[TIS_idx:])
        df_dict["start_codon"].append(DNA_frag[:3])
        prot, has_stop, stop_codon = construct_prot(DNA_frag)
        df_dict["stop_codon"].append(stop_codon)
        df_dict["prot"].append(prot)
        df_dict["TTS_on_transcript"].append(has_stop)
        df_dict["ORF_len"].append(len(prot) * 3)
        TIS_exon = np.sum(TIS_idx >= row.exon_idxs) // 2 + 1
        TIS_exon_idx = TIS_idx - row.exon_idxs[(TIS_exon - 1) * 2]
        if row.strand == b"+":
            TIS_coord = row.exon_coords[(TIS_exon - 1) * 2] + TIS_exon_idx
        else:
            TIS_coord = row.exon_coords[(TIS_exon - 1) * 2 + 1] - TIS_exon_idx
        if has_stop:
            TTS_idx = TIS_idx + df_dict["ORF_len"][-1]
            TTS_pos = TTS_idx + 1
            TTS_exon = np.sum(TTS_idx >= row.exon_idxs) // 2 + 1
            TTS_exon_idx = TTS_idx - row.exon_idxs[(TTS_exon - 1) * 2]
            if row.strand == b"+":
                TTS_coord = row.exon_coords[(TTS_exon - 1) * 2] + TTS_exon_idx
            else:
                TTS_coord = row.exon_coords[(TTS_exon - 1) * 2 + 1] - TTS_exon_idx
        else:
            TTS_coord, TTS_exon, TTS_pos = -1, -1, -1

        df_dict["TIS_coord"].append(TIS_coord)
        df_dict["TIS_exon"].append(TIS_exon)
        df_dict["TTS_pos"].append(TTS_pos)
        df_dict["TTS_exon"].append(TTS_exon)
        df_dict["TTS_coord"].append(TTS_coord)

    df_out = df_out.assign(**df_dict)
    df_out["TIS_idx"] = TIS_idxs
    df_out["correction"] = corrections
    df_out["seqname"] = df_out["contig"]
    df_out["TIS_pos"] = df_out["TIS_idx"] + 1
    df_out["output"] = preds[idxs]
    df_out = df_out.sort_values("output", ascending=False)
    df_out["output_rank"] = np.arange(len(df_out))

    df_out["dist_from_canonical_TIS"] = df_out["TIS_idx"] - df_out["canonical_TIS_idx"]
    df_out.loc[df_out["canonical_TIS_idx"] == -1, "dist_from_canonical_TIS"] = np.nan
    df_out["frame_wrt_canonical_TIS"] = df_out["dist_from_canonical_TIS"] % 3

    if has_seq_output:
        seq_out = [
            f["seq_output"][i][j]
            for i, j in zip(orf_dict["f_idx"], orf_dict["TIS_idx"])
        ]
        orf_dict.update({"seq_output": seq_out})

    if has_ribo_output:
        ribo_subsets = np.array(ribo_id.split(b"&"))
        sparse_reads_set = []
        for subset in ribo_subsets:
            if parallel:
                r = h5py.File(f"{h5_path.split('.h5')[0]}_{subset.decode()}.h5")[
                    "transcript"
                ]
                sparse_reads = h5max.load_sparse(
                    r[f"riboseq/{subset.decode()}/5/"], df_out["f_idx"], to_numpy=False
                )
                r.file.close()
            else:
                sparse_reads = h5max.load_sparse(
                    f[f"riboseq/{subset.decode()}/5/"], df_out["f_idx"], to_numpy=False
                )
            sparse_reads_set.append(sparse_reads)
        sparse_reads = np.add.reduce(sparse_reads_set)
        df_out["reads_in_tr"] = np.array([s.sum() for s in sparse_reads])
        reads_in = []
        reads_out = []
        in_frame_read_perc = []
        for i, (_, row) in tqdm(
            enumerate(df_out.iterrows()),
            total=len(df_out),
            desc=f"{time()}: parsing ribo-seq information ",
        ):
            end_of_ORF_idx = row.TIS_pos + row.ORF_len - 1
            reads_in_ORF = sparse_reads[i][:, row.TIS_pos - 1 : end_of_ORF_idx].sum()
            reads_out_ORF = sparse_reads[i].sum() - reads_in_ORF
            in_frame_reads = sparse_reads[i][
                :, np.arange(row["TIS_pos"] - 1, end_of_ORF_idx, 3)
            ].sum()
            reads_in.append(reads_in_ORF)
            reads_out.append(reads_out_ORF)

            in_frame_read_perc.append(in_frame_reads / max(reads_in_ORF, 1))

        df_out["reads_in_ORF"] = reads_in
        df_out["reads_out_ORF"] = reads_out
        df_out["in_frame_read_perc"] = in_frame_read_perc

    TIS_coords = np.array(f["canonical_TIS_coord"])
    TTS_coords = np.array(f["canonical_TTS_coord"])
    cds_lens = np.array(f["canonical_TTS_idx"]) - np.array(f["canonical_TIS_idx"])
    orf_type = []
    is_cds = []
    for i, row in tqdm(
        df_out.iterrows(),
        total=len(df_out),
        desc=f"{time()}: parsing ORF type information ",
    ):
        TIS_mask = row["TIS_coord"] == TIS_coords
        TTS_mask = row["TTS_coord"] == TTS_coords
        len_mask = row.ORF_len == cds_lens
        is_cds.append(np.logical_and.reduce([TIS_mask, TTS_mask, len_mask]).any())

        if row["canonical_TIS_idx"] != -1:
            if row["canonical_TIS_idx"] == row["TIS_pos"] - 1:
                orf_type.append("annotated CDS")
            elif row["TIS_pos"] > row["canonical_TTS_idx"] + 1:
                orf_type.append("dORF")
            elif row["TTS_pos"] < row["canonical_TIS_idx"] + 1:
                orf_type.append("uORF")
            elif row["TIS_pos"] < row["canonical_TIS_idx"] + 1:
                if row["TTS_pos"] == row["canonical_TTS_idx"] + 1:
                    orf_type.append("N-terminal extension")
                else:
                    orf_type.append("uoORF")
            elif row["TTS_pos"] > row["canonical_TTS_idx"] + 1:
                orf_type.append("doORF")
            else:
                if row["TTS_pos"] == row["canonical_TTS_idx"] + 1:
                    orf_type.append("N-terminal truncation")
                else:
                    orf_type.append("intORF")
        else:
            shares_TIS_coord = row["TIS_coord"] in TIS_coords
            shares_TTS_coord = row["TTS_coord"] in TTS_coords
            if shares_TIS_coord or shares_TTS_coord:
                orf_type.append("CDS variant")
            else:
                orf_type.append("other")
    df_out["ORF_type"] = orf_type
    df_out["ORF_equals_CDS"] = is_cds
    df_out.loc[df_out["tr_biotype"] == b"lncRNA", "ORF_type"] = "lncRNA-ORF"
    # decode strs
    for header in DECODE:
        df_out[header] = df_out[header].str.decode("utf-8")
    df_out["ORF_id"] = df_out["tr_id"] + "_" + df_out["TIS_pos"].astype(str)
    # re-arrange columns
    o_headers = [h for h in OUT_HEADERS if h in df_out.columns]
    df_out = df_out.loc[:, o_headers].sort_values("output_rank")
    # remove duplicates
    if correction and remove_duplicates:
        df_out = df_out.drop_duplicates("ORF_id")
    if exclude_invalid_TTS:
        df_out = df_out[df_out["TTS_on_transcript"]]
    df_out = df_out[df_out["ORF_len"] > min_ORF_len]
    df_out = df_out[df_out["start_codon"].str.contains(start_codons)]
    df_out.to_csv(f"{out_prefix}.csv", index=None)
    f.file.close()

    return df_out


def process_seq_preds(ids, preds, seqs, min_prob):
    df = pd.DataFrame(
        columns=[
            "ID",
            "tr_len",
            "TIS_pos",
            "output",
            "start_codon",
            "TTS_pos",
            "stop_codon",
            "TTS_on_transcript",
            "prot_len",
            "prot_seq",
        ]
    )
    num = 0
    mask = [np.where(pred > min_prob)[0] for pred in preds]
    for i, idxs in enumerate(mask):
        tr = seqs[i]
        for idx in idxs:
            prot_seq, has_stop, stop_codon = construct_prot(tr[idx:])
            TTS_pos = idx + len(prot_seq) * 3
            df.loc[num] = [
                ids[i][0],
                len(tr),
                idx + 1,
                preds[i][idx],
                tr[idx : idx + 3],
                TTS_pos,
                stop_codon,
                has_stop,
                len(prot_seq),
                prot_seq,
            ]
            num += 1
    return df


def create_multiqc_reports(df, out_prefix):
    # Start codons
    output = out_prefix + ".start_codons_mqc.tsv"
    with open(output, "w") as f:
        f.write(RIBOTIE_MQC_HEADER)
        f.write(START_CODON_MQC_HEADER)
    df.start_codon.value_counts().to_csv(output, sep="\t", header=False, mode="a")

    # Transcript biotypes
    output = out_prefix + ".biotypes_variant_mqc.tsv"
    with open(output, "w") as f:
        f.write(RIBOTIE_MQC_HEADER)
        f.write(BIOTYPE_VARIANT_MQC_HEADER)
    orf_biotypes = pd.Series(index=ORF_BIOTYPE_ORDER, data=0)
    counts = df[df.ORF_type == "CDS variant"].tr_biotype.value_counts()
    orf_biotypes[counts.index] = counts
    orf_biotypes.to_csv(output, sep="\t", header=False, mode="a")

    # ORF types
    output = out_prefix + ".ORF_types_mqc.tsv"
    with open(output, "w") as f:
        f.write(RIBOTIE_MQC_HEADER)
        f.write(ORF_TYPE_MQC_HEADER)
    orf_types = pd.Series(index=ORF_TYPE_ORDER, data=0)
    counts = df.ORF_type.value_counts()
    orf_types[counts.index] = counts
    orf_types.to_csv(output, sep="\t", header=False, mode="a")

    # ORF lengths
    # output = out_prefix + ".ORF_lens_mqc.tsv"
    # ax = df.ORF_len.apply(lambda x: np.log(x)).plot.kde()
    # x, y = np.exp(ax.lines[-1].get_xdata()), ax.lines[-1].get_ydata()
    # with open(output, "w") as f:
    #     f.write(RIBOTIE_MQC_HEADER)
    #     f.write(ORF_LEN_MQC_HEADER)
    #     for x_, y_ in zip(x, y):
    #         f.write(f"{x_}\t{y_}\n")

    return


def csv_to_gtf(h5_path, df, out_prefix, exclude_annotated=False):
    """convert RiboTIE result table to GTF"""
    if exclude_annotated:
        df = df.filter(pl.col("ORF_type") != "annotated CDS")
    df = df.fill_null("NA")
    df = df.sort("tr_id")
    f = h5py.File(h5_path, "r")
    f_ids = np.array(f["transcript/id"])
    # fast id mapping
    xsorted = np.argsort(f_ids)
    pred_to_h5_args = xsorted[np.searchsorted(f_ids[xsorted], df["tr_id"])]
    # obtain exons
    exons_coords = np.array(f["transcript/exon_coords"])[pred_to_h5_args]
    f.close()
    gff_parts = []
    for tis, stop_codon_start, strand, exon_coord in zip(
        df["TIS_coord"], df["TTS_coord"], df["strand"], exons_coords
    ):
        start_codon_stop = find_distant_exon_coord(tis, 2, strand, exon_coord)
        start_parts, start_exons = transcript_region_to_exons(
            tis, start_codon_stop, strand, exon_coord
        )
        # acquire cds stop coord from stop codon coord.
        if stop_codon_start != -1:
            stop_codon_stop = find_distant_exon_coord(
                stop_codon_start, 2, strand, exon_coord
            )
            stop_parts, stop_exons = transcript_region_to_exons(
                stop_codon_start, stop_codon_stop, strand, exon_coord
            )
            tts = find_distant_exon_coord(stop_codon_start, -1, strand, exon_coord)
        else:
            stop_parts, stop_exons = np.empty(start_parts.shape), np.empty(
                start_exons.shape
            )
            tts = -1
        cds_parts, cds_exons = transcript_region_to_exons(tis, tts, strand, exon_coord)
        tr_coord = np.array([exon_coord[0], exon_coord[-1]])
        exons = np.arange(1, len(exon_coord) // 2 + 1)
        coords_packed = np.vstack(
            [
                tr_coord.reshape(-1, 2),
                exon_coord.reshape(-1, 2),
                start_parts.reshape(-1, 2),
                cds_parts.reshape(-1, 2),
                stop_parts.reshape(-1, 2),
            ]
        ).astype(int)
        exons_packed = np.hstack(
            [[-1], exons, start_exons, cds_exons, stop_exons]
        ).reshape(-1, 1)
        features_packed = np.hstack(
            [
                np.full(1, "transcript"),
                np.full(len(exons), "exon"),
                np.full(len(start_exons), "start_codon"),
                np.full(len(cds_exons), "CDS"),
                np.full(len(stop_exons), "stop_codon"),
            ]
        ).reshape(-1, 1)
        gff_parts.append(np.hstack([coords_packed, exons_packed, features_packed]))
    gtf_lines = []
    for i, row in enumerate(df.iter_rows(named=True)):
        for start, stop, exon, feature in gff_parts[i]:
            property_list = [
                f'gene_id "{row["gene_id"]}',
                f'transcript_id "{row["ORF_id"]}',
                f'gene_name "{row["gene_name"]}',
                f'transcript_biotype "{row["tr_biotype"]}',
                f'tag "{row["tr_tag"]}',
                f'transcript_support_level "{row["tr_support_lvl"]}',
            ]
            if feature not in ["transcript"]:
                property_list.insert(
                    3,
                    f'exon_number "{exon}',
                )

            if feature not in ["transcript", "exon"]:
                property_list.insert(
                    3,
                    f'ORF_id "{row["ORF_id"]}", model_output "{row["output"]}"; ORF_type "{row["ORF_type"]}"; exon_number "{exon}',
                )
            properties = '"; '.join(property_list)
            gtf_lines.append(
                "\t".join(
                    [
                        row["seqname"],
                        "RiboTIE",
                        feature,
                        start,
                        stop,
                        ".",
                        row["strand"],
                        "0",
                        properties + '";\n',
                    ]
                )
            )
    with open(f"{out_prefix}.gtf", "w") as f:
        for line in gtf_lines:
            f.write(line)
