import sys
from datetime import datetime
import numpy as np
import heapq
from transcript_transformer import CDN_PROT_DICT, PROT_IDX_DICT, DNA_IDX_DICT


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
        if cdn in CDN_PROT_DICT.keys():
            string += CDN_PROT_DICT[cdn]
        else:
            string += "X"

    return string, has_stop, stop_codon


def derive_exon_number():
    # Parse the transcript ID from the "attribute" column
    # Adjust the parsing based on the specific formatting within your GTF file
    # Example assumes something like: 'transcript_id "some_id";'
    get_transcript_id = pl.col("attribute").str.extract(r'transcript_id "([^"]+)"', 1)
    exons_df = exons_df.with_columns(transcript_id=get_transcript_id)

    # Group by transcript and strand, then sort and enumerate exon numbers
    exons_numbered_df = exons_df.sort(["transcript_id", "start"]).with_columns(
        exon_number=pl.col("transcript_id").cumcount().over(["transcript_id", "strand"])
    )

    # Depending on strand, adjust the exon number appropriately
    # The above example increments exon numbers from 0; adjust starting value as needed
    # `.cumcount()` starts from 0, incrementing for each row in the group. To start from 1, add 1.
    exons_numbered_df = exons_numbered_df.with_columns(
        exon_number=pl.when(pl.col("strand") == "+")
        .then(pl.col("exon_number") + 1)
        .otherwise(
            pl.col("exon_number") + 1
        )  # If you ever need reverse order, adjust logic
    )


def DNA2vec(dna_seq, seq_dict=DNA_IDX_DICT):
    dna_vec = np.zeros(len(dna_seq), dtype=int)
    for idx in np.arange(len(dna_seq)):
        dna_vec[idx] = seq_dict[dna_seq[idx]]

    return dna_vec


def prot2vec(prot_seq, prot_dict=PROT_IDX_DICT):
    prot_vec = np.zeros(len(prot_seq), dtype=int)
    for idx in np.arange(len(prot_seq)):
        prot_vec[idx] = prot_dict[prot_seq[idx]]

    return list(prot_vec)


def listify(array):
    return [list(a) for a in array]


def time():
    return datetime.now().strftime("%H:%M:%S %m-%d ")


def vec2DNA(vec, np_dict=np.array(["A", "T", "C", "G", "N"])):
    return "".join(np_dict[vec])


def vec2prot(vec, np_dict=np.array(list(PROT_IDX_DICT.keys()))):
    return "".join(np_dict[vec])


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
            f" to {(1/test_chunks)*100:.1f}% of full data"
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
                f" {(1/val_chunks)*100:.1f}% of train/val data in fold {fold_i}"
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
        start_coord (int) start coordinate
        stop_coord (int): stop coordinate, NOT start of stop codon for CDSs
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
        assert (
            start_coord <= stop_coord
        ), f"start coordinate {start_coord} must be smaller than stop coordinate {stop_coord}"
        exon_idx_start = (
            max(i for i, exon in enumerate(exons) if start_coord >= exon) // 2 * 2
        )
        exon_idx_stop = (
            max(i for i, exon in enumerate(exons) if stop_coord >= exon) // 2 + 1
        ) * 2
        if exon_idx_stop > len(exons) - 1:
            exon_idx_stop = None
        genome_parts = exons[exon_idx_start:exon_idx_stop].copy()
        genome_parts[0], genome_parts[-1] = start_coord, stop_coord
    else:
        assert (
            start_coord >= stop_coord
        ), f"start coordinate {start_coord} must be larger than stop coordinate {stop_coord}"
        exon_idx_start = (
            max(i for i, exon in enumerate(exons) if start_coord <= exon) - 1
        )
        exon_idx_stop = max(i for i, exon in enumerate(exons) if stop_coord <= exon) + 1
        if exon_idx_stop > len(exons) - 1:
            exon_idx_stop = None
        genome_parts = exons[exon_idx_start:exon_idx_stop].copy()
        if len(genome_parts) < 2:
            genome_parts[0], genome_parts[-1] = stop_coord, start_coord
        else:
            genome_parts[1], genome_parts[-2] = start_coord, stop_coord
    if exon_idx_stop is None:
        exon_idx_stop = len(exons) + 1
    exon_numbers = list(range(exon_idx_start // 2, exon_idx_stop // 2))

    return list(genome_parts), [num + 1 for num in exon_numbers]


def check_genomic_order(coords_flat, strand):
    """
    Checks the ordering of exon genomic coordinates based on GTF
    convention and strand.

    For positive strand (+), exons should be ordered by increasing start coordinate.
    For negative strand (-), exons should be ordered by decreasing start coordinate.

    Args:
        coords_flat (list | np.ndarray): A flat list of genomic coordinates in the format
                            [start1, end1, start2, end2, ...].
        strand (str): The strand of the transcript, either '+' or '-'.

    Raises:
        ValueError: If the coordinates are not ordered as expected.
    """

    # --- Input Validation ---
    assert isinstance(
        coords_flat, (list, np.ndarray)
    ), "Error: coords_flat must be a list or np.array."
    assert (
        len(coords_flat) % 2 == 0
    ), "Error: coords_flat must contain an even number of coordinates (start and end pairs)."
    assert strand in ["+", "-"], "Error: strand must be either '+' or '-'."

    # --- Convert flat list to list of exon pairs ---
    exon_pairs = []
    for i in range(0, len(coords_flat), 2):
        start = coords_flat[i]
        end = coords_flat[i + 1]
        if start > end:
            raise ValueError(
                f"Error: Found start coordinate ({start}) greater than end coordinate ({end}) for an exon. GTF standard requires start <= end."
            )
        exon_pairs.append([start, end])

    # --- Check ordering based on strand ---
    if strand == "+":
        # Ensure exons are ordered by increasing start coordinate
        for i in range(len(exon_pairs) - 1):
            if exon_pairs[i][0] > exon_pairs[i + 1][0]:
                raise ValueError(
                    f"Error: Exons are not ordered by increasing start coordinate for positive strand. Found {exon_pairs[i]} before {exon_pairs[i + 1]}."
                )
    else:  # strand == '-'
        # Ensure exons are ordered by decreasing start coordinate
        for i in range(len(exon_pairs) - 1):
            if exon_pairs[i][0] < exon_pairs[i + 1][0]:
                raise ValueError(
                    f"Error: Exons are not ordered by decreasing start coordinate for negative strand. Found {exon_pairs[i]} before {exon_pairs[i + 1]}."
                )


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
        exons (list | np.array): list of exon bound coordinates following gtf file conventions.
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


def get_str2str_idx_map(source, dest):
    xsorted = np.argsort(dest)
    return xsorted[np.searchsorted(dest[xsorted], source)]


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
