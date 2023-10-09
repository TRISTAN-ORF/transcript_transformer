import os
import sys
import numpy as np
import yaml
import itertools
import h5py
from importlib import resources as impresources

from transcript_transformer.transcript_transformer import train
from transcript_transformer.argparser import Parser, parse_config_file
from transcript_transformer.pretrained import riboformer_models
from transcript_transformer.data import process_data
from transcript_transformer.processing import construct_output_table


def parse_args():
    parser = Parser(description="Run Ribo-former", stage="train")
    parser.add_data_args()
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="path prefix to output the results to.",
    )
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
        "--no_correction",
        action="store_false",
        help="don't correct to nearest in-frame ATG, see --distance to adjust distance",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=6,
        help="number of codons to search up- and downstream for an ATG (see also --no_correction)",
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
    parser.add_train_loading_args(pretrain=True)
    args = parser.parse_args(sys.argv[1:])
    args = parse_config_file(args)
    if args.out_prefix is None:
        args.out_prefix = os.path.splitext(args.input_config)[0]
    assert ~args.results and ~args.data, (
        "cannot only do processing of data and results, disable either"
        " --data_process or --result_process"
    )
    args.mlm, args.mask_frac, args.rand_frac = False, False, False

    return args


def load_args(path, args):
    with open(path, "r") as fh:
        input_config = yaml.safe_load(fh)
    args.__dict__.update(input_config)

    return args


def main():
    args = parse_args()
    args = load_args((impresources.files(riboformer_models) / "50perc_06_23.yml"), args)
    if not args.results:
        f = process_data(args)
    assert args.use_ribo, "No ribosome data specified."
    if not (args.data or args.results):
        for i, fold in args.folds.items():
            args.__dict__.update(fold)
            callback_path = (
                impresources.files(riboformer_models) / f"{args.transfer_checkpoint}"
            )
            print(f"--> Loading model: {callback_path}")
            args.transfer_checkpoint = callback_path
            out = train(args, predict=True, enable_model_summary=False)
            preds = list(itertools.chain(*[o[0] for o in out]))
            targets = list(
                itertools.chain(*[[t.astype(bool) for t in o[1]] for o in out])
            )
            ids = list(itertools.chain(*[o[2] for o in out]))
            np.save(
                f"{args.out_prefix}_out_f{i}.npy",
                np.array([ids, preds, targets], dtype=object).T,
            )
        out = np.vstack(
            [
                np.load(f"{args.out_prefix}_out_f{i}.npy", allow_pickle=True)
                for i in args.folds.keys()
            ]
        )
        [os.remove(f"{args.out_prefix}_out_f{i}.npy") for i in args.folds.keys()]
        np.save(f"{args.out_prefix}_out.npy", out)

    if not args.data:
        f = h5py.File(args.h5_path, "r")["transcript"]
        out = np.load(f"{args.out_prefix}_out.npy", allow_pickle=True)
        construct_output_table(
            f,
            out,
            args.out_prefix,
            args.factor,
            args.prob_cutoff,
            ~args.no_correction,
            args.distance,
        )


if __name__ == "__main__":
    main()
