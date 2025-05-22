import os
import sys
import numpy as np
import yaml
import h5py
from importlib.resources import files
from typing import cast
from copy import deepcopy

from .transcript_transformer import train, predict
from .argparser import Parser
from .data import process_seq_data, process_ribo_data
from .processing import construct_output_table, csv_to_gtf, create_multiqc_reports
from .util_functions import (
    find_optimal_folds,
    mv_ckpt_to_out_dir,
    prtime,
    load_args,
    merge_outputs,
)


def parse_args():
    parser = Parser(description="Run RiboTIE", stage="train", tool="ribotie")
    run_parser = parser.add_run_args()
    run_parser.add_argument(
        "--pretrain",
        action="store_true",
        help="pretrain model using all available samples. Highly recommended for riboformer"
        " if no suitable pre-trained model is not available (e.g., when applied on new species)",
    )
    parser.add_data_args()
    pr_parser = parser.add_processing_args()
    pr_parser.add_argument(
        "--no_correction",
        action="store_true",
        help="Don't correct to nearest in-frame ATG, see --distance to adjust distance.",
    )
    pr_parser.add_argument(
        "--distance",
        type=int,
        default=9,
        help="Number of codons to search up- and downstream for an ATG (see also --no_correction).",
    )
    parser.add_comp_args()
    parser.add_training_args()
    parser.add_train_loading_args()
    default_config = files("transcript_transformer.configs").joinpath("defaults.tt.yml")
    default_config = os.fspath(cast(os.PathLike, default_config))
    args = parser.parse_arguments(sys.argv[1:], [default_config])
    if args.out_prefix is None:
        args.out_prefix = f"{os.path.splitext(args.conf[0])[0]}"

    return args


def main():
    prtime("Running RiboTIE tool\n")
    args = parse_args()
    assert args.use_ribo, "No ribosome data specified."

    # --- Data loading ---
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
        args.overwrite_data,
        args.parallel,
        args.low_memory,
    )

    if args.data:
        prtime("End of pipeline (omit --data to run full pipeline)...", "\n")
        return 0

    # --- Determine folds and contigs if relevant ---
    if args.pretrain and (args.folds is None):
        prtime("Determining fold allocations...", "\n")
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
            args.folds = find_optimal_folds(contig_lens, args.test_frac, args.val_frac)

    # --- Pre-training ---
    if args.pretrain and args.missing_models:
        prtime("Pretraining model: training models on collection of all samples", "\n")
        args.transfer_checkpoint = None
        args.metrics = []
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
            args.folds = find_optimal_folds(contig_lens, test=0.5, val=0.2)
        for i, fold in args.folds.items():
            args_set = deepcopy(args)
            # update args.train, args.val, and args.test
            args_set.__dict__.update(fold)
            # set output path
            args_set.out_prefix = "_".join([args.out_prefix, "pretrain", f"f{i}"])
            # train model
            prtime(f"Pre-training model on Fold {i} ...", "\n")
            trainer, model = train(
                args_set, test_model=False, enable_model_summary=False
            )
            # predict TIS locations
            predict(args_set, trainer=trainer, model=model, postprocess=False)
            # saving model
            mv_ckpt_to_out_dir(trainer, f"{args_set.out_prefix}.rt")
            rel_path = os.path.basename(args_set.out_prefix)
            args.folds[i]["transfer_checkpoint"] = f"{rel_path}.rt.ckpt"
        prtime(f"Merging predictions for '{args.out_prefix}_pretrain.npy'...", "\n")
        merge_outputs(f"{args.out_prefix}_pretrain", args.folds.keys())
        # remove independent fold outputs
        [os.remove(f"{args.out_prefix}_f{i}.npy") for i in args.folds.keys()]
        args.folds[0]["test"] = []
        save_dict = {
            "pretrained_model": {"folds": args.folds},
            "patience": 1,
            "lr": 0.0008,
        }
        with open(f"{args.out_prefix}_pretrain_params.yml", "w+") as file_handle:
            yaml.dump(save_dict, file_handle, default_flow_style=False)
    elif not args.missing_models:
        prtime("Pretraining model: training models on collection of all samples", "\n")
        print(
            f"\t -- Found '{args.out_prefix}_params.yml', Skipping RiboTIE pre-training step",
        )

    # --- Fine-tuning Or Prediction ---
    if not args.pretrain:
        for group, ribo_ids in args.grouped_ribo_ids.items():
            prtime(f"Evaluating {group}...", "\n\n")
            # alter configurations per group of samples
            args_set = deepcopy(args)
            args_set.grouped_ribo_ids = {group: ribo_ids}
            args_set.cond["grouped"] = {group: args.cond["grouped"][group]}

            model_file = f"{args.out_prefix}_{group}_params.yml"
            result_file = f"{args.out_prefix}_{group}.npy"
            has_model_output = os.path.isfile(result_file)
            has_model_file = os.path.isfile(model_file)
            # if predicting from trained model, model params is not expected
            if "trained_model" in args:
                keep_model = True
            else:
                keep_model = has_model_file and (not args.overwrite_models)
            keep_preds = has_model_output and (not args.overwrite_preds)
            if keep_model and keep_preds:
                print(f"\t -- RiboTIE output already present: {result_file}")
                continue
            elif (not keep_preds) and "trained_model" in args:
                print(f"\t -- Using listed RiboTIE model")
                finetune = False
            elif not keep_preds and keep_model:
                print(f"\t -- Loading trained RiboTIE model: {model_file}")
                args = load_args(model_file, args)
                finetune = False
            elif "pretrained_model" in args:
                print(f"\t -- Using listed PRE-trained RiboTIE model")
                finetune = True
                folds = args.pretrained_model["folds"]
            else:
                print(f"\t -- Using default pre-trained RiboTIE model (Human)")
                model_params = files(
                    "transcript_transformer.pretrained.rt_models"
                ).joinpath("50perc_06_23.rt.yml")
                args = load_args(model_params, args)
                finetune = True
                args.model_dir = files("transcript_transformer.pretrained.rt_models")
            if finetune:
                folds = args.pretrained_model["folds"]
            else:
                folds = args.trained_model["folds"]
            args.folds = deepcopy(folds)
            # iterate models
            for i, fold in folds.items():
                args_set.__dict__.update(fold)
                args_set.transfer_checkpoint = os.path.join(
                    args.model_dir, args_set.transfer_checkpoint
                )
                args_set.out_prefix = f"{args.out_prefix}_{group}_f{i}"
                if finetune:
                    prtime(f"Finetuning model for {group} — Fold {i} ...", "\n")
                    print(f"\t -- Loaded model: {args_set.transfer_checkpoint}...")
                    trainer, model = train(
                        args_set, test_model=False, enable_model_summary=False
                    )
                    mv_ckpt_to_out_dir(trainer, args_set.out_prefix)
                    # set output path
                    rel_path = os.path.basename(args_set.out_prefix)
                    args.folds[i]["transfer_checkpoint"] = f"{rel_path}.rt.ckpt"
                    prtime(f"Predicting samples for {group} — Fold {i} ...", "\n")
                    predict(args_set, trainer=trainer, model=model, postprocess=False)
                else:
                    prtime(f"Predicting samples for {group} — Fold {i} ...", "\n")
                    print(f"\t -- Loaded model: {args_set.transfer_checkpoint}...")
                    predict(args_set, postprocess=False)
            prtime(f"Merging predictions to '{args.out_prefix}_{group}.npy'...", "\n")
            merge_outputs(f"{args.out_prefix}_{group}", folds.keys())
            # Save params file
            if finetune:
                args.folds[0]["test"] = []
                save_dict = {
                    "trained_model": {"folds": args.folds},
                    "patience": 1,
                    "lr": 0.0008,
                }
                with open(f"{args.out_prefix}_{group}_params.yml", "w+") as file_handle:
                    yaml.dump(save_dict, file_handle, default_flow_style=False)

    if args.pretrain:
        output_sets = ["pretrain"]
    else:
        output_sets = [str(k) for k in args.grouped_ribo_ids.keys()]
    for output in output_sets:
        out = np.load(f"{args.out_prefix}_{output}.npy", allow_pickle=True)
        out_prefix = f"{args.out_prefix}_{output}"
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
            return_ORF_coords=args.return_ORF_coords,
        )
        if df is not None:
            ids = ["ribotie_all", "ribotie"]
            names = ["RiboTIE_redundant", "RiboTIE"]
            paths = [out_prefix + ".redundant", out_prefix]
            multiqc_path = os.path.join(os.path.dirname(args.out_prefix), "multiqc")
            os.makedirs(multiqc_path, exist_ok=True)
            for df, id, name, path in zip([df, df_filt], ids, names, paths):
                csv_to_gtf(args.h5_path, df, path, "RiboTIE", args.exclude_annotated)
                out = os.path.join(multiqc_path, os.path.basename(path))
                create_multiqc_reports(df, out, id, name)
        else:
            print("No positive predictions found")

        if args.pretrain:
            print(
                f"""\n
                !!! 
                Do not use the pre-trained predictions as is.
                RiboTIE is meant to fine-tune on individual samples after pre-training.
                Run RiboTIE without the --pretrain flag and with newly created yml file, e.g.,
                'ribotie {' '.join(args.conf)} {args.out_prefix}_pretrain_params.yml'.
                !!!
                """
            )


if __name__ == "__main__":
    main()
