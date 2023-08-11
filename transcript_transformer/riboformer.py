import os
import sys
import numpy as np
import yaml
import itertools
from importlib import resources as impresources

from transcript_transformer.transcript_transformer import train
from transcript_transformer.argparser import Parser, parse_config_file
from transcript_transformer.pretrained import riboformer_models
from transcript_transformer.data import process_data


def parse_args():
    parser = Parser(description="Run Ribo-former", stage="train")
    parser.add_data_args()
    parser.add_argument(
        "--out", type=str, default="rf", help="path (prefix) of output files"
    )
    parser.add_argument(
        "--data_process",
        action="store_true",
        help="only perform data processing step",
    )
    parser.add_comp_args()
    parser.add_training_args()
    parser.add_train_loading_args(pretrain=True)
    args = parser.parse_args(sys.argv[1:])
    args = parse_config_file(args)
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
    process_data(args)
    for i, fold in args.folds.items():
        args.__dict__.update(fold)
        callback_path = (
            impresources.files(riboformer_models) / f"{args.transfer_checkpoint}"
        )
        print(f"--> Loading model: {callback_path}")
        args.transfer_checkpoint = callback_path
        out = train(args, predict=True, enable_model_summary=False)
        preds = list(itertools.chain(*[o[0] for o in out]))
        targets = list(itertools.chain(*[o[1] for o in out]))
        ids = list(itertools.chain(*[o[2] for o in out]))
        np.save(
            f"{args.out}_out_f{i}.npy", np.array([ids, preds, targets], dtype=object).T
        )
    out = np.vstack(
        [
            np.load(f"{args.out}_out_f{i}.npy", allow_pickle=True)
            for i in args.folds.keys()
        ]
    )
    [os.remove(f"{args.out}_out_f{i}.npy") for i in args.folds.keys()]
    np.save(f"{args.out}_out.npy", out)


if __name__ == "__main__":
    main()
