import os
import sys
import numpy as np
import yaml
import h5py
from importlib import resources as impresources
from copy import deepcopy

from .transcript_transformer import train, predict
from .argparser import Parser
from .pretrained import ribotie_models
from . import configs
from .data import process_seq_data, process_ribo_data
from .processing import construct_output_table, csv_to_gtf, create_multiqc_reports
from .util_functions import define_folds


def parse_args():
    parser = Parser(description="Run RiboTIE", stage="train")
    data_parser = parser.add_data_args()
    data_parser.add_argument(
        "--prob_cutoff",
        type=float,
        default=0.125,
        help="Determines the minimum model output score required for model "
        "predictions to be included in the result table.",
    )
    data_parser.add_argument(
        "--start_codons",
        type=str,
        default=".*TG$",
        help="Valid start codons to include in result table. Uses regex string.",
    )
    data_parser.add_argument(
        "--min_ORF_len",
        type=int,
        default="15",
        help="Minimum nucleotide length of predicted translated open reading frames.",
    )
    data_parser.add_argument(
        "--include_invalid_TTS",
        action="store_true",
        help="Include translated ORF predictions with no TTS on transcript.",
    )
    data_parser.add_argument(
        "--no_correction",
        action="store_true",
        help="Don't correct to nearest in-frame ATG, see --distance to adjust distance.",
    )
    data_parser.add_argument(
        "--distance",
        type=int,
        default=9,
        help="Number of codons to search up- and downstream for an ATG (see also --no_correction).",
    )
    data_parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Don't remove duplicate ORFs resulting from the correction step.",
    )
    data_parser.add_argument(
        "--exclude_annotated",
        action="store_true",
        help="Exclude annotated CDS regions in generated GTF file containing predicted translated ORFs.",
    )
    data_parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="Don't apply filtering of top predictions",
    )
    # data_parser.add_argument(
    #     "--pretrained_model",
    #     type=json.loads,
    #     default="{}",
    #     help="dictionary (nested) containing pretrained model info. Automatically generated within a dedicated "
    #     "config when creating a pretrained model (--pretrained). Uses recommended pretrained model if left blank",
    # )
    data_parser.add_argument(
        "--data",
        action="store_true",
        help="Only perform pre-processing of data.",
    )
    data_parser.add_argument(
        "--results",
        action="store_true",
        help="Only perform processing of model predictions.",
    )
    parser.add_comp_args()
    parser.add_training_args()
    parser.add_train_loading_args(pretrain=True, auto=True)
    default_config = f"{impresources.files(configs) / 'ribotie_defaults.yml'}"
    args = parser.parse_arguments(sys.argv[1:], [default_config])
    if args.out_prefix is None:
        args.out_prefix = f"{os.path.splitext(args.conf[0])[0]}_"
    assert ~args.results and ~args.data, (
        "cannot only do processing of data and results, disable either"
        " --data_process or --result_process"
    )
    args.mlm, args.mask_frac, args.rand_frac = False, False, False
    args.use_seq = False
    args.input_type = "hdf5"
    return args


def load_args(path, args):
    with open(path, "r") as fh:
        input_config = yaml.safe_load(fh)
    args.__dict__.update(input_config)

    return args


def main():
    args = parse_args()
    assert args.use_ribo, "No ribosome data specified."
    if not args.results:
        process_seq_data(
            args.h5_path,
            args.gtf_path,
            args.fa_path,
            args.backup_path,
            not args.no_backup,
        )
        process_ribo_data(
            args.h5_path,
            args.ribo_paths,
            args.overwrite,
            args.parallel,
            args.low_memory,
        )
    # Pre-training
    if args.pretrain and (not args.data):
        args.transfer_checkpoint = None
        args.metrics = []
        print(
            f"--> Pretraining model: Combining all samples in one training, validation and test set"
        )
        f = h5py.File(args.h5_path, "r")["transcript"]
        contigs = np.array(f["seqname"])
        tr_lens = np.array(f["transcript_len"])
        f.file.close()
        contig_set = np.unique(contigs)
        if args.folds == None:
            contig_lens = {}
            # determine nt count per seqname
            for contig in contig_set:
                mask = contigs == contig
                contig_lens[contig] = sum(tr_lens[mask])
            args.folds = define_folds(contig_lens, test=0.5, val=0.2)
            print(args.folds)
        if not args.results:
            for i, fold in args.folds.items():
                args_set = deepcopy(args)
                # update args.train, args.val, and args.test
                args_set.__dict__.update(fold)
                # set output path
                args_set.out_prefix = args.out_prefix + f"pretrain_f{i}"
                # train model
                trainer, model = train(
                    args_set, test_model=False, enable_model_summary=False
                )
                # predict TIS locations
                predict(args_set, trainer=trainer, model=model, postprocess=False)
                # saving model
                ckpt_path = os.path.join(trainer.logger.log_dir, "checkpoints")
                weights_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
                os.replace(weights_path, f"{args_set.out_prefix}.ckpt")
                args.folds[i]["transfer_checkpoint"] = f"{args_set.out_prefix}.ckpt"
            args.folds[0]["test"] = []

            with open(f"{args.out_prefix}pretrain.yml", "w+") as f:
                yaml.dump(
                    {
                        "pretrained_model": {"folds": args.folds},
                        "patience": 1,
                        "lr": 0.0008,
                    },
                    f,
                    default_flow_style=False,
                )

    # Fine-tuning
    if not (args.data or args.results or args.pretrain):
        if "pretrained_model" not in args:
            args = load_args(
                (impresources.files(ribotie_models) / "50perc_06_23.yml"), args
            )
            args.model_dir = str(impresources.files(ribotie_models)._paths[0])
        for group, ribo_ids in args.grouped_ribo_ids.items():
            # alter configurations per group of samples
            args_set = deepcopy(args)
            args_set.grouped_ribo_ids = {group: ribo_ids}
            args_set.cond["grouped"] = {group: args.cond["grouped"][group]}
            for j, fold in args_set.pretrained_model["folds"].items():
                args_set.__dict__.update(fold)
                args_set.transfer_checkpoint = os.path.join(
                    args.model_dir, args_set.transfer_checkpoint
                )
                print(f"--> Loading model: {args_set.transfer_checkpoint}...")
                print(f"--> Finetuning model for {group}...")
                trainer, model = train(
                    args_set, test_model=False, enable_model_summary=False
                )
                print(f"--> Predicting samples for {group}...")
                # set output path
                args_set.out_prefix = f"{args.out_prefix}{group}_f{j}"
                predict(args_set, trainer=trainer, model=model, postprocess=False)
            prefix = f"{args.out_prefix}{group}"
            merge_outputs(prefix, args.pretrained_model["folds"].keys())

    if not args.data:
        if args.pretrain:
            output_sets = [f"pretrain_f{i}" for i, fold in args.folds.items()]
        else:
            output_sets = [str(k) for k in args.grouped_ribo_ids.keys()]

        for output in output_sets:
            out = np.load(f"{args.out_prefix}{output}.npy", allow_pickle=True)
            out_prefix = f"{args.out_prefix}{output}"
            df, df_filt = construct_output_table(
                h5_path=args.h5_path,
                out_prefix=out_prefix,
                prob_cutoff=args.prob_cutoff,
                correction=not args.no_correction,
                dist=args.distance,
                start_codons=args.start_codons,
                min_ORF_len=args.min_ORF_len,
                remove_duplicates=not args.keep_duplicates,
                exclude_invalid_TTS=not args.include_invalid_TTS,
                ribo_output=out,
                grouped_ribo_ids=args.grouped_ribo_ids,
                parallel=args.parallel,
            )
            if df is not None:
                ids = ["ribotie_all", "ribotie"]
                names = ["RiboTIE_unfiltered", "RiboTIE"]
                paths = [out_prefix + ".unfiltered", out_prefix]
                multiqc_path = os.path.join(os.path.dirname(args.out_prefix), "multiqc")
                os.makedirs(multiqc_path, exist_ok=True)
                for df, id, name, path in zip([df, df_filt], ids, names, paths):
                    csv_to_gtf(
                        args.h5_path, df, path, "RiboTIE", args.exclude_annotated
                    )
                    out = os.path.join(multiqc_path, os.path.basename(path))
                    create_multiqc_reports(df, out, id, name)
            else:
                print("No positive predictions found")


def merge_outputs(prefix, keys):
    out = np.vstack([np.load(f"{prefix}_f{i}.npy", allow_pickle=True) for i in keys])
    np.save(f"{prefix}.npy", out)
    [os.remove(f"{prefix}_f{i}.npy") for i in keys]


if __name__ == "__main__":
    main()
