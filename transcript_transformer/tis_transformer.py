import os
import sys
import numpy as np
import yaml
import h5py
from importlib import resources as impresources
import heapq
from argparse import Namespace

from transcript_transformer.transcript_transformer import train, predict
from transcript_transformer.argparser import Parser, parse_config_file
from transcript_transformer.pretrained import tis_transformer_models
from transcript_transformer.data import process_seq_data
from transcript_transformer.processing import construct_output_table


def parse_args():
    parser = Parser(description="Run Ribo-former", stage="train")
    parser.add_data_args()
    parser.add_argument(
        "--factor",
        type=float,
        default=1,
        help="Determines the number of model predictions in the result table."
        "This factor is multiplied to the number of canonical "
        "TISs present on evaluated transcripts.",
    )
    parser.add_argument(
        "--prob_cutoff",
        type=float,
        default=0.03,
        help="Determines the minimum model output score required for model "
        "predictions to be included in the result table.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="only perform pre-processing of data",
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="only perform processing of model predictions",
    )
    parser.add_comp_args()
    parser.add_training_args()
    parser.add_train_loading_args(pretrain=False)
    parser.add_evaluation_args()
    parser.add_architecture_args()
    args = load_args(
        (impresources.files(tis_transformer_models) / "default_config.yml")
    )
    args.__dict__.update(**vars(parser.parse_args(sys.argv[1:])))
    args = parse_config_file(args)
    if args.out_prefix is None:
        args.out_prefix = os.path.splitext(args.input_config)[0]
    assert ~args.results and ~args.data, (
        "cannot only do processing of data and results, disable either"
        " --data_process or --result_process"
    )
    args.mlm, args.mask_frac, args.rand_frac = False, False, False
    # remove riboformer specific properties
    args.use_seq = True
    args.use_ribo = False
    args.cond["grouped"] = [{}]
    return args


def load_args(path):
    with open(path, "r") as fh:
        input_config = yaml.safe_load(fh)

    return Namespace(**input_config)


def main():
    args = parse_args()
    prefix = f"{args.out_prefix}_seq"
    if not args.results:
        process_seq_data(
            args.h5_path, args.gtf_path, args.fa_path, args.backup_path, ~args.no_backup
        )
    if not (args.data or args.results):
        args.use_seq = True
        args.use_ribo = False
        args.input_type = "config"
        f = h5py.File(args.h5_path, "r")["transcript"]
        contigs = np.array(f["contig"])
        tr_lens = np.array(f["tr_len"])
        f.close()
        contig_set = np.unique(contigs)
        contig_lens = {}
        for contig in contig_set:
            mask = contigs == contig
            contig_lens[contig] = sum(tr_lens[mask])
        folds = define_folds(contig_lens, 0.2, 0.2)
        for i, fold in folds.items():
            args.__dict__.update(fold)
            trainer, model = train(args, test_model=False, enable_model_summary=False)
            args.out_prefix = f"{prefix}_f{i}"
            predict(args, trainer=trainer, model=model, postprocess=False)
        merge_outputs(prefix, folds.keys())

        f = h5py.File(args.h5_path, "a")
        grp = f["transcript"]
        out = np.load(f"{prefix}.npy", allow_pickle=True)
        construct_output_table(
            grp, out, prefix, args.factor, args.prob_cutoff, ribo=args.use_ribo
        )
        f_tr_ids = np.array(grp["id"])
        xsorted = np.argsort(f_tr_ids)
        tr_ids = np.hstack([o[0] for o in out])

        pred_to_h5_args = xsorted[np.searchsorted(f_tr_ids[xsorted], tr_ids)]
        pred_arr = np.empty(shape=(len(f_tr_ids),), dtype=object)
        pred_arr.fill(np.array([], dtype=np.float32))
        for idx, (_, pred, _) in zip(pred_to_h5_args, out):
            pred_arr[idx] = pred
        dtype = h5py.vlen_dtype(np.dtype("float32"))
        if "seq_output" in grp.keys():
            print("--> Overwriting results in local h5 database...")
            del grp["seq_output"]
        else:
            print("--> Writing results to local h5 database...")
        grp.create_dataset("seq_output", data=pred_arr, dtype=dtype)
        f.close()
        if not args.no_backup:
            if not args.backup_path:
                args.backup_path = os.path.splitext(args.gtf_path)[0] + ".h5"
            if os.path.isfile(args.backup_path):
                f = h5py.File(args.backup_path, "a")
                grp = f["transcript"]
                if "seq_output" in grp.keys():
                    print("--> Overwriting results in backup h5 database...")
                    del grp["seq_output"]
                else:
                    print("--> Writing results to backup h5 database...")
                grp.create_dataset("seq_output", data=pred_arr, dtype=dtype)
                f.close()
    if not args.data:
        f = h5py.File(args.h5_path, "r")
        construct_output_table(
            f["transcript"], f"{args.out_prefix}_seq", args.factor, args.prob_cutoff
        )
        f.close()


def merge_outputs(prefix, keys):
    out = np.vstack([np.load(f"{prefix}_f{i}.npy", allow_pickle=True) for i in keys])
    np.save(f"{prefix}.npy", out)
    [os.remove(f"{prefix}_f{i}.npy") for i in keys]


def divide_seqnames(seqname_count_dict, num_chunks):
    arr = np.array(list(seqname_count_dict.values()))
    labels = np.array(list(seqname_count_dict.keys()))
    idxs = np.argsort(arr)[::-1]
    arr = np.sort(arr)[::-1]
    heap = [(0, idx) for idx in range(num_chunks)]
    heapq.heapify(heap)
    sets_v = {}
    sets_k = {}
    for i in range(num_chunks):
        sets_v[i] = []
        sets_k[i] = []
    arr_idx = 0
    while arr_idx < len(arr):
        set_sum, set_idx = heapq.heappop(heap)
        sets_k[set_idx].append(labels[idxs][arr_idx])
        sets_v[set_idx].append(arr[idxs][arr_idx])
        set_sum += arr[arr_idx]
        heapq.heappush(heap, (set_sum, set_idx))
        arr_idx += 1

    folds = zip(sets_k.values(), sets_v.values())
    return {i: {x: y for x, y in zip(k, v)} for i, (k, v) in enumerate(folds)}


def define_folds(seqn_size_dict, test=0.2, val=0.2):
    test_chunks = int(np.ceil(1 / test))
    val_chunks_set = int(np.ceil(1 / val))
    contig_set = np.array(list(seqn_size_dict.keys()))
    if len(contig_set) < test_chunks:
        test_chunks = len(contig_set)
        print(
            f"!-> Not enough seqnames to divide data, increasing test set"
            f" to {(1/test_chunks):.2f}% of full data"
        )
    groups = divide_seqnames(seqn_size_dict, test_chunks)
    folds = {}
    for fold_i, group in groups.items():
        mask = np.isin(contig_set, list(group.keys()))
        test_set = contig_set[mask]
        tr_val_set = contig_set[~mask]
        tr_val_lens = {k: v for k, v in seqn_size_dict.items() if k in tr_val_set}
        if len(tr_val_lens) < val_chunks_set:
            val_chunks = len(tr_val_lens)
            print(
                f"!-> Not enough seqnames to divide data, increasing val set to"
                f" {(1/val_chunks):.2f}% of train/val data in fold {fold_i}"
            )
        else:
            val_chunks = val_chunks_set
        tr_val_groups = divide_seqnames(tr_val_lens, val_chunks)
        # Find group that is closest to queried partition
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
        folds[fold_i] = {"train": train_set, "val": val_set, "test": test_set}

    return folds


if __name__ == "__main__":
    main()
