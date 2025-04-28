import os
import sys
import numpy as np
import h5py
import yaml
from importlib import resources as impresources

from .transcript_transformer import train, predict
from .argparser import Parser
from .util_functions import define_folds
from . import configs
from .data import process_seq_data
from .processing import construct_output_table, csv_to_gtf, create_multiqc_reports


def parse_args():
    parser = Parser(description="Run TIS Transformer", stage="train")
    data_parser = parser.add_data_args()
    data_parser.add_argument(
        "--prob_cutoff",
        type=float,
        default=0.03,
        help="Determines the minimum model output score required for model "
        "predictions to be included in the result table.",
    )
    data_parser.add_argument(
        "--data",
        action="store_true",
        help="only perform pre-processing of data",
    )
    data_parser.add_argument(
        "--results",
        action="store_true",
        help="only perform processing of model predictions",
    )
    data_parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Determines the fraction of the training set to be used for "
        "validation. The remaining fraction is used for training.",
    )
    data_parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Determines the fraction of the full set to be used for "
        "testing. The remaining fraction is used for training/validation.",
    )
    data_parser.add_argument(
        "--exclude_annotated",
        action="store_true",
        help="Exclude annotated CDS regions in generated GTF file containing predicted translated ORFs.",
    )
    parser.add_comp_args()
    parser.add_training_args()
    parser.add_train_loading_args(pretrain=False)
    parser.add_evaluation_args()
    parser.add_architecture_args()
    default_config = f"{impresources.files(configs) / 'tis_transformer_defaults.yml'}"
    args = parser.parse_arguments(sys.argv[1:], [default_config])
    if args.out_prefix is None:
        print(args.conf[0])
        args.out_prefix = f"{os.path.splitext(args.h5_path)[0]}_"
    assert ~args.results and ~args.data, (
        "cannot only do processing of data and results, disable either"
        " --data_process or --result_process"
    )
    args.mlm, args.mask_frac, args.rand_frac = False, False, False
    # remove riboformer specific properties
    return args


def main():
    args = parse_args()
    prefix = f"{args.out_prefix}"
    if not args.results:
        process_seq_data(
            args.h5_path,
            args.gtf_path,
            args.fa_path,
            args.backup_path,
            not args.no_backup,
        )
    # Training
    if not (args.data or args.results or args.predict):
        # determine optimal allocation of seqnames to train/val/test set
        f = h5py.File(args.h5_path, "r")["transcript"]
        contigs = np.array(f["seqname"])
        tr_lens = np.array(f["transcript_len"])
        f.file.close()
        # determine nt count per seqname
        contig_set = np.unique(contigs)
        contig_lens = {}
        for contig in contig_set:
            mask = contigs == contig
            contig_lens[contig] = sum(tr_lens[mask])
        # assign seqnames to train/val/test set
        args.folds = define_folds(contig_lens, args.test_frac, args.val_frac)
        for i, fold in args.folds.items():
            args.__dict__.update(fold)
            # set output path
            args.out_prefix = f"{prefix}_f{i}"
            # train model
            trainer, model = train(args, test_model=False, enable_model_summary=False)
            # predict TIS locations
            predict(args, trainer=trainer, model=model, postprocess=False)
            # saving model
            ckpt_path = os.path.join(trainer.logger.log_dir, "checkpoints")
            weights_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
            os.replace(weights_path, f"{args.out_prefix}.ckpt")
            args.folds[i]["transfer_checkpoint"] = f"{args.out_prefix}.ckpt"
        args.folds[0]["test"] = []

        with open(f"{args.out_prefix}parameters.yml", "w+") as f:
            yaml.dump(
                {
                    "trained_model": {"folds": args.folds},
                },
                f,
                default_flow_style=False,
            )

        # merge all predictions into one file
        merge_outputs(prefix, args.folds.keys())

        # sort and transfer predictions to h5 file
        f = h5py.File(args.h5_path, "a")
        grp = f["transcript"]
        f_tr_ids = np.array(grp["transcript_id"])
        # load predictions
        out = np.load(f"{prefix}.npy", allow_pickle=True)
        tr_ids = np.hstack([o[0] for o in out])
        tr_ids = [s.split(b"|")[1] for s in tr_ids]
        # fast re-order predictions to hdf5 ids
        xsorted = np.argsort(f_tr_ids)
        pred_to_h5_args = xsorted[np.searchsorted(f_tr_ids[xsorted], tr_ids)]
        # create empty arrays to store predictions
        pred_arr = np.empty(shape=(len(f_tr_ids),), dtype=object)
        pred_arr.fill(np.array([], dtype=np.float32))
        for idx, (_, pred, _) in zip(pred_to_h5_args, out):
            pred_arr[idx] = pred
        dtype = h5py.vlen_dtype(np.dtype("float32"))
        if "tis_transformer_score" in grp.keys():
            print("--> Overwriting results in local h5 database...")
            del grp["tis_transformer_score"]
        else:
            print("--> Writing results to local h5 database...")
        grp.create_dataset("tis_transformer_score", data=pred_arr, dtype=dtype)
        f.close()
        if not args.no_backup:
            if not args.backup_path:
                args.backup_path = os.path.splitext(args.gtf_path)[0] + ".h5"
            if os.path.isfile(args.backup_path):
                f = h5py.File(args.backup_path, "a")
                grp = f["transcript"]
                if "tis_transformer_score" in grp.keys():
                    print("--> Overwriting results in backup h5 database...")
                    del grp["tis_transformer_score"]
                else:
                    print("--> Writing results to backup h5 database...")
                grp.create_dataset("tis_transformer_score", data=pred_arr, dtype=dtype)
                f.close()

    # Result Processing
    if not args.data:
        df, df_filt = construct_output_table(args.h5_path, prefix, args.prob_cutoff)
        if df is not None:
            names = ["TIS Transformer Unfiltered", "TIS Transformer"]
            paths = [prefix + ".unfiltered", prefix]
            multiqc_path = os.path.join(os.path.dirname(args.out_prefix), "multiqc")
            os.makedirs(multiqc_path, exist_ok=True)
            for df, name, path in zip([df, df_filt], names, paths):
                csv_to_gtf(
                    args.h5_path, df, path, "TIS_Transformer", args.exclude_annotated
                )
                out = os.path.join(multiqc_path, os.path.basename(path))
                create_multiqc_reports(df, out, "tis_transformer", name)


def merge_outputs(prefix, keys):
    out = np.vstack([np.load(f"{prefix}_f{i}.npy", allow_pickle=True) for i in keys])
    np.save(f"{prefix}.npy", out)
    [os.remove(f"{prefix}_f{i}.npy") for i in keys]


if __name__ == "__main__":
    main()
