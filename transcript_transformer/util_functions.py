from datetime import datetime
import numpy as np
import heapq


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


def DNA2vec(dna_seq):
    seq_dict = {"A": 0, "T": 1, "U": 1, "C": 2, "G": 3, "N": 4}
    dna_vec = np.zeros(len(dna_seq), dtype=int)
    for idx in np.arange(len(dna_seq)):
        dna_vec[idx] = seq_dict[dna_seq[idx]]

    return dna_vec


def time():
    return datetime.now().strftime("%H:%M:%S %m-%d ")


def vec2DNA(tr_seq, np_dict=np.array(["A", "T", "C", "G", "N"])):
    return "".join(np_dict[tr_seq])


def divide_keys_by_size(size_dict, num_chunks):
    """divide units in parts that resemble in size

    Args:
        size_dict (dict): size of each unit
        num_chunks (int): number of parts

    Returns:
        dict: nested dict with unit (key) and size (value) for each part (outer).
    """
    sizes = np.array(list(size_dict.values()))
    labels = np.array(list(size_dict.keys()))
    idxs = np.argsort(sizes)[::-1]
    sorted_labels = labels[idxs]
    sorted_sizes = np.sort(sizes)[::-1]
    # create list of tuples to store size of parts (size, part_id)
    heap = [(0, idx) for idx in range(num_chunks)]
    heapq.heapify(heap)
    sets_v = {}
    sets_k = {}
    # init keys and values
    for i in range(num_chunks):
        sets_k[i] = []
        sets_v[i] = []
    arr_idx = 0
    # for each element (large to small), add to part
    while arr_idx < len(sorted_sizes):
        # pop the smallest item from the list and add unit
        set_sum, set_idx = heapq.heappop(heap)
        sets_k[set_idx].append(sorted_labels[arr_idx])
        sets_v[set_idx].append(sorted_sizes[arr_idx])
        set_sum += sorted_sizes[arr_idx]
        heapq.heappush(heap, (set_sum, set_idx))
        arr_idx += 1
    folds = zip(sets_k.values(), sets_v.values())

    return {i: {x: y for x, y in zip(k, v)} for i, (k, v) in enumerate(folds)}


def define_folds(seqn_size_dict, test=0.2, val=0.2):
    """Finds closest possible folds for given seqnames.

    Args:
        seqn_size_dict (dict): nucleotide size for each seqname
        test (float, optional): fraction of the test set. Defaults to 0.2.
        val (float, optional): fraction of the validation set (excl. test fraction).
            Defaults to 0.2.

    Returns:
        dict: nested dictionary containing seqnames of "train", "val", "test"(inner)
        set for each fold (outer).
    """

    # find number of parts required to allow test set fraction
    test_chunks = int(np.ceil(1 / test))
    val_chunks_set = int(np.ceil(1 / val))
    seqname_set = np.array(list(seqn_size_dict.keys()))
    if len(seqname_set) < test_chunks:
        test_chunks = len(seqname_set)
        print(
            f"!-> Not enough seqnames to divide data, increasing test set"
            f" to {(1/test_chunks):.2f}% of full data"
        )
    # group seqnames in number of parts that will allow listed test set fraction
    groups = divide_keys_by_size(seqn_size_dict, test_chunks)
    folds = {}
    for fold_i, group in groups.items():
        mask = np.isin(seqname_set, list(group.keys()))
        test_set = seqname_set[mask]
        tr_val_set = seqname_set[~mask]
        tr_val_lens = {k: v for k, v in seqn_size_dict.items() if k in tr_val_set}
        if len(tr_val_lens) < val_chunks_set:
            val_chunks = len(tr_val_lens)
            print(
                f"!-> Not enough seqnames to divide data, increasing val set to"
                f" {(1/val_chunks):.2f}% of train/val data in fold {fold_i}"
            )
        else:
            val_chunks = val_chunks_set
        # Find number of parts required to allow val set fraction
        tr_val_groups = divide_keys_by_size(tr_val_lens, val_chunks)
        tot_count = sum(tr_val_lens.values())
        val_counts = np.empty(val_chunks)
        for i, val in enumerate(tr_val_groups.values()):
            np.array(list(val.values()))
            val_sum = sum(val.values())
            val_counts[i] = val_sum / (tot_count - val_sum)
        group_idx = np.argmin(abs(val_counts - 1 / val_chunks))
        val_mask = np.isin(tr_val_set, list(tr_val_groups[group_idx].keys()))
        val_set, train_set = tr_val_set[val_mask], tr_val_set[~val_mask]
        tr = [t.decode() for t in train_set]
        val = [t.decode() for t in val_set]
        test = [t.decode() for t in test_set]
        print(f"\tFold {fold_i}: train: {tr}, val: {val}, test: {test}")
        folds[fold_i] = {
            "train": tr,
            "val": val,
            "test": test,
        }

    return folds


def transcript_region_to_exons(
    start_coord, stop_coord, strand, exons, region_length=None
):
    """Return exons connecting two gene coordinates. Both start and
    stop coordinates must exist on exons.

    Args:
        start_coord (str): start coordinate
        stop_coord (str): stop coordinate, NOT start of stop codon for CDSs
        strand (str): strand, either + or -
        exons (list): list of exon bound coordinates following gtf file conventions.
            E.g. positive strand: [1 2 4 5] negative strand: [4 5 1 2]
        region_length (int): length of region, defaults to None, stop_coord is ignored if filled

    Returns:
        list: list of bound coordinates following gtf file conventions.
            E.g. [{start_coord} 2 4 {stop_coord}]
    """
    pos_strand = strand == "+"
    if stop_coord == -1:
        if pos_strand:
            stop_coord = exons[-1]
        elif len(exons) == 2:
            stop_coord = exons[0]
        else:
            stop_coord = exons[-2]
    if pos_strand:
        exon_idx_start = np.where(start_coord >= exons)[0][-1] // 2 * 2
        exon_idx_stop = (np.where(stop_coord >= exons)[0][-1] // 2 + 1) * 2
        if exon_idx_stop > len(exons) - 1:
            exon_idx_stop = None
        genome_parts = exons[exon_idx_start:exon_idx_stop].copy()
        genome_parts[[0, -1]] = start_coord, stop_coord
    else:
        exon_idx_start = np.where(start_coord <= exons)[0][-1] - 1
        exon_idx_stop = np.where(stop_coord <= exons)[0][-1] + 1
        if exon_idx_stop > len(exons) - 1:
            exon_idx_stop = None
        genome_parts = exons[exon_idx_start:exon_idx_stop].copy()
        if len(genome_parts) < 2:
            genome_parts[[0, -1]] = stop_coord, start_coord
        else:
            genome_parts[[1, -2]] = start_coord, stop_coord
    if exon_idx_stop is None:
        exon_idx_stop = len(exons) + 1
    exon_numbers = np.arange(exon_idx_start // 2, exon_idx_stop // 2)

    return genome_parts, exon_numbers + 1


def get_exon_dist_map(tr_regions, strand):
    """Get map that relates distance of exons on a processed transcript to those of
    genomic coordinates.

    Args:
        tr_regions (list): list of transcript region coordinates split according to
            exons and following gtf file conventions.
            E.g. positive strand: [1 2 4 5] negative strand: [4 5 1 2]

    Returns:
        np.array: map where index of vector equate to index of transcript and array
            values equate genomic coordinates. note that index is 0-coordinate system while
            the genomic coordinates follow the 1-coordinate system.
    """
    if strand == "+":
        return np.hstack(
            [np.arange(k, l + 1) for k, l in np.array(tr_regions).reshape(-1, 2)]
        )
    else:
        return np.hstack(
            [np.arange(l, k - 1, -1) for k, l in np.array(tr_regions).reshape(-1, 2)]
        )


def get_exon_lengths(exons):
    """Get length of exons.

    Args:
        exons (list): list of exon bound coordinates following gtf file conventions.
            E.g. positive strand: [1 2 4 5] negative strand: [4 5 1 2]

    Returns:
        np.array: list of exon lengths
    """
    exon_bounds = np.array(exons).reshape(-1, 2)
    exon_lens = abs(exon_bounds[:, 1] - exon_bounds[:, 0]) + 1

    return exon_lens.ravel()


def find_distant_exon_coord(ref_coord, distance, strand, exons):
    """find genome exon coordinate from reference coord and distance. Both coordinates
    exist on exons

    Args:
        ref_coord (str): reference coordinate
        distance (int): distance to reference coordinate, positive distances equate to
            distances downstream of the processed transcript.
        strand (str): strand, either "+" or "-".
        exons (list): list of exon bound coordinates following gtf file conventions.
            E.g. positive strand: [1 2 4 5] negative strand: [4 5 1 2]
        region_length (int): length of region, defaults to None, stop_coord is ignored if filled

    Returns:
        list: list of distant coordinate following gtf file conventions.
            E.g.
    """
    exon_map = get_exon_dist_map(exons, strand)

    exon_map_ref_idx = np.where(exon_map == ref_coord)[0][0]
    exon_map_new_idx = exon_map_ref_idx + distance
    if (exon_map_new_idx > 0) and (exon_map_new_idx < len(exon_map)):
        dist_coord = exon_map[exon_map_new_idx]
    else:
        dist_coord = -1

    return dist_coord
