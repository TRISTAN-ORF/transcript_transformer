import os
import sys
import numpy as np
import yaml
import h5py
from importlib import resources as impresources
from copy import deepcopy

from .transcript_transformer import train, predict
from .argparser import Parser, parse_config_file
from .pretrained import riboformer_models
from .data import process_seq_data, process_ribo_data
from .processing import construct_output_table, csv_to_gtf, create_multiqc_reports


def parse_args():
    parser = Parser(description="Run Ribo-former", stage="train")
    parser.add_data_args()
    parser.add_argument(
        "--prob_cutoff",
        type=float,
        default=0.15,
        help="Determines the minimum model output score required for model "
        "predictions to be included in the result table.",
    )
    parser.add_argument(
        "--start_codons",
        type=str,
        default=".*TG$",
        help="Valid start codons to include in result table. Uses regex string.",
    )
    parser.add_argument(
        "--min_ORF_len",
        type=int,
        default="15",
        help="Minimum nucleotide length of predicted translated open reading frames.",
    )
    parser.add_argument(
        "--include_invalid_TTS",
        action="store_true",
        help="Include translated ORF predictions with no TTS on transcript.",
    )
    parser.add_argument(
        "--no_correction",
        action="store_false",
        help="Don't correct to nearest in-frame ATG, see --distance to adjust distance.",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=9,
        help="Number of codons to search up- and downstream for an ATG (see also --no_correction).",
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Don't remove duplicate ORFs resulting from the correction step.",
    )
    parser.add_argument(
        "--exclude_annotated",
        action="store_true",
        help="Include annotated CDS regions in generated GTF file containing predicted translated ORFs.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Only perform pre-processing of data.",
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Only perform processing of model predictions.",
    )
    parser.add_comp_args()
    parser.add_training_args()
    parser.add_train_loading_args(pretrain=True)
    args = parser.parse_args(sys.argv[1:])
    args = parse_config_file(args)
    if args.out_prefix is None:
        args.out_prefix = f"{os.path.splitext(args.input_config)[0]}_"
    assert ~args.results and ~args.data, (
        "cannot only do processing of data and results, disable either"
        " --data_process or --result_process"
    )
    args.mlm, args.mask_frac, args.rand_frac = False, False, False
    args.use_seq = False
    return args


def load_args(path, args):
    with open(path, "r") as fh:
        input_config = yaml.safe_load(fh)
    args.__dict__.update(input_config)

    return args


def main():
    args = parse_args()
    args = load_args((impresources.files(riboformer_models) / "50perc_06_23.yml"), args)
    assert args.use_ribo, "No ribosome data specified."
    if not args.results:
        process_seq_data(
            args.h5_path, args.gtf_path, args.fa_path, args.backup_path, ~args.no_backup
        )
        process_ribo_data(
            args.h5_path, args.ribo_paths, args.overwrite, args.low_memory
        )
    if not (args.data or args.results):
        args.input_type = "config"
        for i, ribo_set in enumerate(args.ribo_ids):
            args_set = deepcopy(args)
            args_set.ribo_ids = [ribo_set]
            args_set.cond["grouped"] = [args.cond["grouped"][i]]
            ribo_set_str = "&".join(ribo_set)
            for j, fold in args_set.folds.items():
                args_set.__dict__.update(fold)
                callback_path = (
                    impresources.files(riboformer_models)
                    / f"{args_set.transfer_checkpoint}"
                )
                print(f"--> Loading model: {callback_path}...")
                args_set.transfer_checkpoint = callback_path
                print(f"--> Finetuning model for {ribo_set_str}...")
                trainer, model = train(
                    args_set, test_model=False, enable_model_summary=False
                )
                print(f"--> Predicting samples for {ribo_set_str}...")
                args_set.out_prefix = f"{args.out_prefix}{ribo_set_str}_f{j}"
                predict(args_set, trainer=trainer, model=model, postprocess=False)

            ribo_set_str = "&".join(ribo_set)
            prefix = f"{args.out_prefix}{ribo_set_str}"
            merge_outputs(prefix, args.folds.keys())

    if not args.data:
        f = h5py.File(args.h5_path, "r")
        for ribo_set in args.ribo_ids:

            ribo_set_str = "&".join(ribo_set)
            out = np.load(f"{args.out_prefix}{ribo_set_str}.npy", allow_pickle=True)
            out_prefix = f"{args.out_prefix}{ribo_set_str}"
            df = construct_output_table(
                f=f["transcript"],
                out_prefix=out_prefix,
                prob_cutoff=args.prob_cutoff,
                correction=not args.no_correction,
                dist=args.distance,
                start_codons=args.start_codons,
                min_ORF_len=args.min_ORF_len,
                remove_duplicates=not args.keep_duplicates,
                exclude_invalid_TTS=not args.include_invalid_TTS,
                ribo=out,
            )
            if df is not None:
                csv_to_gtf(f, df, out_prefix, args.exclude_annotated)
                os.makedirs(f"{args.out_prefix}/multiqc", exist_ok=True)
                create_multiqc_reports(df, f"{args.out_prefix}/multiqc/{ribo_set_str}")
        f.close()


def merge_outputs(prefix, keys):
    out = np.vstack([np.load(f"{prefix}_f{i}.npy", allow_pickle=True) for i in keys])
    np.save(f"{prefix}.npy", out)
    [os.remove(f"{prefix}_f{i}.npy") for i in keys]


if __name__ == "__main__":
    main()
