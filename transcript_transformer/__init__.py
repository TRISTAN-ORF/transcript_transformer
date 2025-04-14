# Define global variables

CDN_PROT_DICT = {
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
    "NNN": "X",
}

PROT_IDX_DICT = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "E": 5,
    "Q": 6,
    "G": 7,
    "H": 8,
    "O": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "_": 21,
    "X": 22,
}

DNA_IDX_DICT = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3,
    "N": 4,
}

IDX_PROT_DICT = {v: k for k, v in PROT_IDX_DICT.items()}
IDX_DNA_DICT = {v: k for k, v in DNA_IDX_DICT.items()}

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

STANDARD_HEADERS = [
    "CDS_coords",
    "CDS_idxs",
    "canonical_TIS_coord",
    "canonical_TIS_exon",
    "canonical_TIS_idx",
    "canonical_LTS_coord",
    "canonical_LTS_idx",
    "canonical_TTS_coord",
    "canonical_TTS_idx",
    "canonical_protein_seq",
    "exon_coords",
    "exon_idxs",
    "gene_id",
    "has_annotated_start_codon",
    "has_annotated_stop_codon",
    "seq",
    "seqname",
    "source",
    "strand",
    "transcript_id",
    "transcript_len",
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

RENAME_HEADERS = {
    "has_annotated_start_codon": "CDS_has_annotated_start_codon",
    "has_annotated_stop_codon": "CDS_has_annotated_stop_codon",
}

STANDARD_OUT_HEADERS = [
    "seqname",
    "ORF_id",
    "ORF_len",
    "transcript_id",
    "transcript_len",
    "start_codon",
    "stop_codon",
    "strand",
    "ORF_type",
    "TIS_pos",
    "TTS_pos",
    "has_CDS_clones",
    "has_CDS_TIS",
    "has_CDS_TTS",
    "shared_in_frame_CDS_frac",
    "dist_from_canonical_TIS",
    "frame_wrt_canonical_TIS",
    "TTS_on_transcript",
    "TIS_coord",
    "TIS_exon",
    "TTS_coord",
    "TTS_exon",
    "LTS_coord",
    "LTS_exon",
    "gene_id",
    "canonical_TIS_coord",
    "canonical_TIS_pos",
    "canonical_LTS_coord",
    "canonical_LTS_pos",
    "canonical_TTS_coord",
    "canonical_TTS_pos",
    "has_annotated_start_codon",
    "has_annotated_stop_codon",
    "protein_seq",
]

RIBO_OUT_HEADERS = [
    "correction",
    "reads_in_transcript",
    "reads_in_ORF",
    "reads_in_frame_frac",
    "reads_5UTR",
    "reads_3UTR",
    "reads_coverage_frac",
    "reads_entropy",
    "reads_skew",
]


RIBOTIE_MQC_HEADER = """
# parent_id: {id}
# parent_name: {name}
# parent_description: "Overview of open reading frames called as translating by RiboTIE"
# """

START_CODON_MQC_HEADER = """
# id: 'ribotie_start_codon_counts_{id}' 
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
# id: 'ribotie_biotype_counts_variant_{id}'
# section_name: 'Transcript Biotypes (varRNA-ORF)'
# description: "Transcript biotypes of varRNA-ORFs called by RiboTIE"
# plot_type: 'bargraph'
# anchor: 'transcript_biotype_variant_counts'
# pconfig:
#     id: "transcript_biotype_counts_variant_plot"
#     title: "RiboTIE: varRNA-ORFs Transcript Biotypes"
#     xlab: "# ORFs"
#     cpswitch_counts_label: "Number of ORFs"
"""

ORF_TYPE_MQC_HEADER = """
# id: 'ribotie_orftype_counts_{id}'
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
# id: 'ribotie_orflen_hist_{id}'
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
    "C-terminal truncation",
    "C-terminal extension",
    "uORF",
    "uoORF",
    "dORF",
    "doORF",
    "intORF",
    "lncRNA-ORF",
    "varRNA-ORF",
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
