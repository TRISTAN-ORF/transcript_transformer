import numpy as np
import h5max
from datetime import datetime
from tqdm import tqdm
import pandas as pd

cdn_prot_dict = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "_",
    "TAG": "_",
    "TGC": "C",
    "TGT": "C",
    "TGA": "_",
    "TGG": "W",
    "NNN": "_",
}


headers = [
    "id",
    "contig",
    "biotype",
    "strand",
    "canonical_TIS_coord",
    "canonical_TIS_exon_idx",
    "canonical_TIS_idx",
    "canonical_TTS_coord",
    "canonical_TTS_idx",
    "canonical_prot_id",
    "exon_coords",
    "exon_idxs",
    "gene_id",
    "gene_name",
    "support_lvl",
    "tag",
    "tr_len",
]


out_headers = [
    "seqname",
    "id",
    "TIS_pos",
    "output",
    "output_rank",
    "seq_output",
    "start_codon",
    "stop_codon",
    "ORF_len",
    "TTS_pos",
    "TTS_on_transcript",
    "reads_per_base",
    "rpb_in_ORF",
    "rpb_out_ORF",
    "in_frame_read_perc",
    "ORF_type",
    "biotype",
    "support_lvl",
    "tag",
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
    "canonical_prot_id",
    "prot",
]

decode = [
    "seqname",
    "id",
    "biotype",
    "support_lvl",
    "tag",
    "strand",
    "gene_id",
    "gene_name",
    "canonical_prot_id",
]


def vec2DNA(tr_seq, np_dict=np.array(["A", "T", "C", "G", "N"])):
    return "".join(np_dict[tr_seq])


def time():
    return datetime.now().strftime("%H:%M:%S %m-%d ")


def construct_prot(seq):
    stop_cds = ["TAG", "TGA", "TAA"]
    sh_cds = np.array([seq[n : n + 3] for n in range(0, len(seq) - 2, 3)])
    stop_site_pos = np.where(np.isin(sh_cds, stop_cds))[0]
    if len(stop_site_pos) > 0:
        has_stop = True
        stop_site = stop_site_pos[0]
        stop_codon = sh_cds[stop_site]
        cdn_seq = sh_cds[:stop_site]
    else:
        has_stop = False
        stop_codon = None
        cdn_seq = sh_cds

    string = ""
    for cdn in cdn_seq:
        if "N" in cdn:
            string += "_"
        else:
            string += cdn_prot_dict[cdn]

    return string, has_stop, stop_codon


def construct_output_table(
    f,
    out_prefix,
    factor=0.8,
    prob_cutoff=0.05,
    correction=False,
    dist=9,
    ribo=None,
):
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
        canon_TIS_sum = sum([o[2].sum() for o in ribo])

    else:
        canon_TIS_sum = np.hstack(f["tis"]).sum()
        mask = [len(o) > 0 for o in np.array(f["seq_output"])]
        preds = np.hstack(np.array(f["seq_output"]))
        pred_to_h5_args = np.where(mask)[0]
    num_top_results = int(canon_TIS_sum * factor)
    # map pred ids to database id
    num_high_preds = (preds > prob_cutoff).sum()

    k = max(min(num_top_results, num_high_preds), 100)
    lens = np.array(f["tr_len"])[pred_to_h5_args]
    cum_lens = np.cumsum(np.insert(lens, 0, 0))
    idxs = np.argpartition(preds, -k)[-k:]

    data_dict = {"f_idx": [], "TIS_idx": []}
    for idx in idxs:
        idx_tr = np.where(cum_lens > idx)[0][0] - 1
        data_dict["TIS_idx"].append(idx - cum_lens[idx_tr])
        data_dict["f_idx"].append(pred_to_h5_args[idx_tr])
    if has_seq_output:
        seq_out = [
            f["seq_output"][i][j]
            for i, j in zip(data_dict["f_idx"], data_dict["TIS_idx"])
        ]
        data_dict.update({"seq_output": seq_out})
    data_dict.update(
        {
            f"{header}": np.array(f[f"{header}"])[data_dict["f_idx"]]
            for header in headers
        }
    )
    data_dict.update(data_dict)
    df_out = pd.DataFrame(data=data_dict)
    df_out["correction"] = np.nan

    data_dict = {
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
        df_out.iterrows(),
        total=len(df_out),
        desc=f"{time()}: parsing ORF information ",
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
        data_dict["start_codon"].append(DNA_frag[:3])
        prot, has_stop, stop_codon = construct_prot(DNA_frag)
        data_dict["stop_codon"].append(stop_codon)
        data_dict["prot"].append(prot)
        data_dict["TTS_on_transcript"].append(has_stop)
        data_dict["ORF_len"].append(len(prot) * 3)
        TIS_exon = np.sum(TIS_idx >= row.exon_idxs) // 2 + 1
        TIS_exon_idx = TIS_idx - row.exon_idxs[(TIS_exon - 1) * 2]
        if row.strand == b"+":
            TIS_coord = row.exon_coords[(TIS_exon - 1) * 2] + TIS_exon_idx
        else:
            TIS_coord = row.exon_coords[(TIS_exon - 1) * 2 + 1] - TIS_exon_idx
        if has_stop:
            TTS_idx = TIS_idx + data_dict["ORF_len"][-1]
            TTS_pos = TTS_idx + 1
            TTS_exon = np.sum(TTS_idx >= row.exon_idxs) // 2 + 1
            TTS_exon_idx = TTS_idx - row.exon_idxs[(TTS_exon - 1) * 2]
            if row.strand == b"+":
                TTS_coord = row.exon_coords[(TTS_exon - 1) * 2] + TTS_exon_idx
            else:
                TTS_coord = row.exon_coords[(TTS_exon - 1) * 2 + 1] - TTS_exon_idx
        else:
            TTS_coord, TTS_exon, TTS_pos = -1, -1, -1

        data_dict["TIS_coord"].append(TIS_coord)
        data_dict["TIS_exon"].append(TIS_exon)
        data_dict["TTS_pos"].append(TTS_pos)
        data_dict["TTS_exon"].append(TTS_exon)
        data_dict["TTS_coord"].append(TTS_coord)

    df_out = df_out.assign(**data_dict)
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

    if has_ribo_output:
        ribo_subsets = np.array(ribo_id.split(b"@"))
        sparse_reads_set = []
        for subset in ribo_subsets:
            sparse_reads = h5max.load_sparse(
                f[f"riboseq/{subset.decode()}/5/"], df_out["f_idx"], to_numpy=False
            )
            sparse_reads_set.append(sparse_reads)
        sparse_reads = np.add.reduce(sparse_reads_set)
        df_out["reads_per_base"] = (
            np.array([s.sum() for s in sparse_reads]) / df_out.tr_len
        )
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

            reads_in.append(reads_in_ORF / max(1, row.ORF_len))
            reads_out.append(reads_out_ORF / max(1, (row.tr_len - row.ORF_len)))
            in_frame_read_perc.append(in_frame_reads / max(reads_in_ORF, 1))

        df_out["rpb_in_ORF"] = reads_in
        df_out["rpb_out_ORF"] = reads_out
        df_out["in_frame_read_perc"] = in_frame_read_perc

    TIS_coords = np.array(f["canonical_TIS_coord"])
    TTS_coords = np.array(f["canonical_TTS_coord"])
    orf_type = []
    for i, row in tqdm(
        df_out.iterrows(),
        total=len(df_out),
        desc=f"{time()}: parsing ORF type information ",
    ):
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
    df_out.loc[df_out["biotype"] == b"lncRNA", "ORF_type"] = "lncRNA-ORF"
    o_headers = [h for h in out_headers if h in df_out.columns]
    df_out = df_out.loc[:, o_headers].sort_values("output_rank")
    for header in decode:
        df_out[header] = df_out[header].str.decode("utf-8")
    df_out.to_csv(f"{out_prefix}.csv", index=None)


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
