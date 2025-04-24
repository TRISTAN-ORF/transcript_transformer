import os
import json
import yaml
import argparse
import importlib
import numpy as np


class Parser(argparse.ArgumentParser):
    def __init__(self, stage="None", **kwargs):
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )
        self.add_argument(
            "conf",
            nargs="*",
            help="paths to YAML or JSON formatted arguments",
        )
        if stage == "train":
            self.add_misc_args()

    def add_data_args(self):
        input_parse = self.add_argument_group("Data processing arguments")
        input_parse.add_argument(
            "--gtf_path", type=str, help="(Required) path to assembly gtf file."
        )
        input_parse.add_argument(
            "--fa_path", type=str, help="(Required) path to assembly reference sequence"
        )
        input_parse.add_argument(
            "--h5_path",
            type=str,
            help="(Required) path used for generating hdf5 file",
        )
        input_parse.add_argument(
            "--ribo_paths",
            type=json.loads,
            help="dictionary containing ribosome sample ID's and path to reads mapped to transcriptome (.bam)."
            " recommended to define in config file",
        )
        input_parse.add_argument(
            "--samples",
            type=str,
            nargs="+",
            help="samples to process. These have to be listed in 'ribo_paths'. Allows merging of reads of  "
            "multiple samples. --samples can also be used to process part of your "
            "database, e.g., in combination with running --parallel",
        )
        input_parse.add_argument(
            "--out_prefix",
            type=str,
            help="output prefix used for all output files, defaults to derivative of 'h5_path'",
        )
        input_parse.add_argument(
            "--cond",
            type=json.loads,
            help="remove transcripts from training based on transcript properties. This does not affect "
            "transcripts in validation/test sets. Currently only supports number of mapped riboseq reads "
            "property. Can filter per dataset. See template.yml file for more examples",
        )
        input_parse.add_argument(
            "--overwrite",
            action="store_true",
            help="overwrite ribo-seq data when id is already present in h5 file",
        )
        input_parse.add_argument(
            "--parallel",
            action="store_true",
            help="create separate hdf5 databases for ribo-seq samples, used when parallalizing "
            "workflows (e.g., with snakemake). This prevents potential read/write"
            " problems that arise when multiple separate processes use the same database",
        )
        input_parse.add_argument(
            "--no_backup",
            action="store_true",
            help="Do not create a backup of the processed assembly in the GTF folder location",
        )
        input_parse.add_argument(
            "--backup_path",
            type=str,
            default=None,
            help="path to backup location (defaults to GTF folder location)",
        )
        input_parse.add_argument(
            "--offsets",
            type=json.loads,
            help="(NOT RECOMMENDED) functionality only exists for benchmarking purposes.",
        )
        input_parse.add_argument(
            "--low_memory",
            action="store_true",
            help="polars setting that trades perfomance for memory-efficiency. use on low-memory devices.",
        )
        input_parse.add_argument(
            "--cores",
            type=int,
            default=8,
            help="number of processor cores used for data processing",
        )

        return input_parse

    def add_comp_args(self):
        comp_parse = self.add_argument_group("Computational resources arguments")
        comp_parse.add_argument(
            "--num_workers", type=int, default=8, help="number of processor cores"
        )
        comp_parse.add_argument(
            "--max_memory",
            type=int,
            default=30000,
            help="Value (GPU vRAM) used to bucket batches based on rough estimates. "
            "Reduce this setting if running out of memory",
        )
        comp_parse.add_argument(
            "--accelerator",
            type=str,
            default="gpu",
            choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"],
            help="computational hardware to apply",
        )
        comp_parse.add_argument(
            "--strategy",
            type=str,
            default="auto",
            help="strategy for multi-gpu computation",
        )
        comp_parse.add_argument(
            "--devices", type=int, default=1, nargs="+", help="GPU device to use"
        )

        return comp_parse

    def add_architecture_args(self):
        tf_parse = self.add_argument_group("Model architecture arguments")
        tf_parse.add_argument(
            "--num_tokens",
            type=int,
            default=8,
            help="number of unique nucleotide input tokens (for sequence input)",
        )
        tf_parse.add_argument(
            "--dim", type=int, default=30, help="dimension of the hidden states"
        )
        tf_parse.add_argument("--depth", type=int, default=6, help="number of layers")
        tf_parse.add_argument(
            "--heads",
            type=int,
            default=6,
            help="number of attention heads in every layer",
        )
        tf_parse.add_argument(
            "--dim_head",
            type=int,
            default=16,
            help="dimension of the attention head matrices",
        )
        tf_parse.add_argument(
            "--nb_features",
            type=int,
            default=80,
            help="number of random features, if not set, will default to (d * log(d)), "
            "where d is the dimension of each head",
        )
        tf_parse.add_argument(
            "--feature_redraw_interval",
            type=int,
            default=1000,
            help="how frequently to redraw the projection matrix",
        )
        tf_parse.add_argument(
            "--no_generalized_attention",
            action="store_false",
            help="applies generalized attention functions",
        )
        tf_parse.add_argument(
            "--reversible",
            action="store_true",
            help="reversible layers, from Reformer paper",
        )
        tf_parse.add_argument(
            "--ff_chunks",
            type=int,
            default=1,
            help="chunk feedforward layer, from Reformer paper",
        )
        tf_parse.add_argument(
            "--use_scalenorm",
            action="store_true",
            help="use scale norm, from 'Transformers without Tears' paper",
        )
        tf_parse.add_argument(
            "--use_rezero",
            action="store_true",
            help="use rezero, from 'Rezero is all you need' paper",
        )
        tf_parse.add_argument(
            "--ff_glu", action="store_true", help="use GLU variant for feedforward"
        )
        tf_parse.add_argument(
            "--emb_dropout", type=float, default=0.1, help="embedding dropout"
        )
        tf_parse.add_argument(
            "--ff_dropout", type=float, default=0.1, help="feedforward dropout"
        )
        tf_parse.add_argument(
            "--attn_dropout", type=float, default=0.1, help="post-attn dropout"
        )
        tf_parse.add_argument(
            "--local_attn_heads",
            type=int,
            default=4,
            help="the amount of heads used for local attention",
        )
        tf_parse.add_argument(
            "--local_window_size",
            type=int,
            default=256,
            help="window size of local attention",
        )

        return tf_parse

    def add_predict_loading_args(self):
        dl_parse = self.add_argument_group("Data loading arguments")
        dl_parse.add_argument(
            "transfer_checkpoint",
            type=str,
            help="Path to checkpoint trained model",
        )
        dl_parse.add_argument(
            "--test",
            type=str,
            nargs="*",
            default=[],
            help="chromosomes from h5 database to predict on",
        )
        dl_parse.add_argument(
            "--min_seq_len",
            type=int,
            default=0,
            help="minimum sequence length of transcripts",
        )
        dl_parse.add_argument(
            "--max_seq_len",
            type=int,
            default=50000,
            help="maximum sequence length of transcripts",
        )
        dl_parse.add_argument(
            "--max_transcripts_per_batch",
            type=int,
            default=2000,
            help="maximum of transcripts per batch",
        )

        return dl_parse

    def add_train_loading_args(self, pretrain=False, auto=False):
        dl_parse = self.add_argument_group("Data loading arguments")
        if not pretrain:
            dl_parse.add_argument(
                "--transfer_checkpoint",
                type=str,
                help="Path to checkpoint pretrained model",
            )
        if not auto:
            dl_parse.add_argument(
                "--train",
                type=str,
                nargs="*",
                default=[],
                help="chromosomes used for training. If not specified, "
                "training is performed on all available chromosomes excluding val/test seqnames",
            )
            dl_parse.add_argument(
                "--val",
                type=str,
                nargs="*",
                default=[],
                help="chromosomes used for validation",
            )
            dl_parse.add_argument(
                "--test",
                type=str,
                nargs="*",
                default=[],
                help="chromosomes used for testing",
            )
        dl_parse.add_argument(
            "--strict_validation",
            action="store_true",
            help="does not apply custom loading filters (see 'cond') defined in config file to validation set",
        )
        dl_parse.add_argument(
            "--leaky_frac",
            type=float,
            default=0.05,
            help="fraction of training samples that escape filtering through "
            "conditions defined in input config file",
        )
        dl_parse.add_argument(
            "--min_seq_len",
            type=int,
            default=0,
            help="minimum sequence length of transcripts",
        )
        dl_parse.add_argument(
            "--max_seq_len",
            type=int,
            default=30000,
            help="maximum sequence length of transcripts",
        )
        dl_parse.add_argument(
            "--max_transcripts_per_batch",
            type=int,
            default=2000,
            help="maximum of transcripts per batch",
        )

        return dl_parse

    def add_training_args(self):
        tr_parse = self.add_argument_group("Model training arguments")
        tr_parse.add_argument(
            "--pretrain",
            action="store_true",
            help="pretrain model using all available samples. Highly recommended for riboformer"
            " if no suitable pre-trained model is not available (e.g., when applied on new species)",
        )
        tr_parse.add_argument(
            "--folds",
            type=json.loads,
            default=None,
            help="only for --pretrain. Recommended to leave empty to automatically detect folds of equal size. "
            "Dictionary containing the seqnames/contigs for the training, validation and test.",
        )
        tr_parse.add_argument(
            "--log_dir",
            type=str,
            default="models",
            help="folder in which training logs and model checkpoints are generated.",
        )
        tr_parse.add_argument(
            "--name",
            type=str,
            default="",
            help="name of the model/run",
        )
        tr_parse.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        tr_parse.add_argument(
            "--decay_rate",
            type=float,
            default=0.96,
            help="multiplicatively decays learning rate for every epoch",
        )
        tr_parse.add_argument(
            "--warmup_steps",
            type=int,
            default=1500,
            help="number of warmup steps at the start of training",
        )
        tr_parse.add_argument(
            "--max_epochs", type=int, default=60, help="maximum epochs of training"
        )
        tr_parse.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Number of epochs required without the validation loss reducing"
            "to stop training",
        )

        return tr_parse

    def add_selfsupervised_args(self):
        ss_parse = self.add_argument_group("Self-supervised training arguments")
        ss_parse.add_argument(
            "--mask_frac",
            type=float,
            default=0.85,
            help="fraction of inputs that are masked",
        )
        ss_parse.add_argument(
            "--rand_frac",
            type=float,
            default=0.10,
            help="fraction of inputs that are randomized",
        )

        return ss_parse

    def add_evaluation_args(self):
        ev_parse = self.add_argument_group("Model evaluation arguments")
        ev_parse.add_argument(
            "--metrics",
            type=str,
            nargs="*",
            default=[],
            choices=["ROC", "PR"],
            help="metrics calculated on the validation and test"
            " set. These can require substantial vRAM and cause OOM crashes",
        )

        return ev_parse

    def add_custom_data_args(self):
        cd_parse = self.add_argument_group("Custom data loading arguments")
        cd_parse.add_argument(
            "input_data",
            type=str,
            metavar="input_data",
            help="input config file, fasta file, or RNA sequence",
        )
        cd_parse.add_argument(
            "input_type",
            type=str,
            default="hdf5",
            metavar="input_type",
            help="type of input",
            choices=["hdf5", "fasta", "RNA"],
        )

        return cd_parse

    # TODO Cleanup
    def add_preds_args(self):
        pr_parse = self.add_argument_group("Model prediction processing arguments")
        pr_parse.add_argument(
            "--min_prob",
            type=float,
            default=0.01,
            help="minimum prediction threshold at which additional information is processed",
        )
        pr_parse.add_argument(
            "--out_prefix",
            type=str,
            default="results",
            help="path (prefix) of output files, ignored if using config input file",
        )

        return pr_parse

    def add_misc_args(self):
        ms_parse = self.add_argument_group("Miscellaneous arguments")
        ms_parse.add_argument(
            "--debug",
            action="store_true",
            help="debug mode disables logging and checkpointing (only for training)",
        )
        ms_parse.add_argument(
            "--version",
            action="version",
            version=importlib.metadata.version("transcript-transformer"),
        )

        return ms_parse

    def parse_arguments(self, argv, configs=[]):
        # update default values (before --help is called)
        model_dir = ""
        for conf in configs:
            with open(conf, "r") as f:
                if conf[-4:] == "json":
                    input_config = json.load(f)
                else:
                    input_config = yaml.safe_load(f)
                if "pretrained_model" in input_config.keys():
                    model_dir = os.path.dirname(os.path.realpath(f.name))
                self.set_defaults(**input_config)
        # parse arguments
        args = self.parse_args(argv)
        # read passed config file
        for conf in args.conf:
            with open(conf, "r") as f:
                if conf[-4:] == "json":
                    input_config = json.load(f)
                else:
                    input_config = yaml.safe_load(f)
                self.set_defaults(**input_config)
        # override config file with bash inputs
        args = self.parse_args()
        args.model_dir = model_dir
        # create output dir if non-existent
        if args.out_prefix:
            os.makedirs(
                os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True
            )
        # backward compatibility
        if "seq" in args:
            args.use_seq = args.seq
        # determine presence of ribo samples
        args.use_ribo = (
            "ribo_paths" in args
            and isinstance(args.ribo_paths, dict)
            and bool(args.ribo_paths)
        )
        if args.h5_path:
            args.h5_path = f"{args.h5_path.split('.h5')[0].split('.hdf5')[0]}.h5"

        if args.use_ribo:
            # by default, the keys are both the group names and individual sample names (first idx of list)
            init_paths = {k: [[k, v]] for k, v in args.ribo_paths.items()}
            # remap ribo_paths based on args.samples
            if args.samples:
                if isinstance(args.samples, list):
                    # Reduce ribo_paths to only include paths for samples in the provided list
                    args.ribo_paths = {
                        sample: init_paths[sample]
                        for sample in args.samples
                        if sample in init_paths
                    }
                elif isinstance(args.samples, dict):
                    # Group ribo_paths based on the provided sample groups
                    new_ribo_paths = {}
                    for group_name, sample_list in args.samples.items():
                        # Ensure value is either a string or a list of strings
                        assert isinstance(sample_list, str) or (
                            isinstance(sample_list, list)
                            and all(isinstance(x, str) for x in sample_list)
                        ), (
                            "Sample when given a study id should be a string or list of str of file id, "
                            "recheck your yaml to make sure it fits the documentation"
                        )
                        new_ribo_paths[group_name] = [
                            init_paths[sample][0]
                            for sample in sample_list
                            if sample in init_paths
                        ]
                    args.ribo_paths = new_ribo_paths
                else:
                    raise ValueError(
                        "Invalid format for args.samples. Expected list or dict."
                    )
            else:
                args.ribo_paths = init_paths
            # Construct {group_id: [sample_id 1 , ..]} dict
            args.grouped_ribo_ids = {
                k: [v[0] for v in vs] for k, vs in args.ribo_paths.items()
            }
        else:
            args.grouped_ribo_ids = {}

        # Construct conditions for data filtering
        conds = {"global": {}}
        conds["grouped"] = {k: {} for k in args.grouped_ribo_ids.keys()}
        conds["global"]["transcript_len"] = lambda x: np.logical_and(
            x > args.min_seq_len, x < args.max_seq_len
        )
        if args.cond is not None:
            # Conditions applied to ribo-seq data
            if "ribo" in args.cond.keys() and args.use_ribo:
                for key, item in args.cond["ribo"].items():
                    # if dictionary, apply conditions to each group of samples
                    if isinstance(item, dict):
                        for id, cond in item.items():
                            # Update the corresponding condition for each matching group
                            for group_name, samples in args.ribo_paths.items():
                                paths = [s[1] for s in samples]
                                if id in paths:
                                    tmp_dict = {f"{key}": lambda x: eval(cond)}
                                    conds["grouped"][group_name].update(tmp_dict)
                    # otherwise, apply conditions to all samples
                    else:
                        # Add the condition to all groups
                        for group_name in args.ribo_paths.keys():
                            tmp_dict = {f"{key}": lambda x: eval(item)}
                            conds["grouped"][group_name].update(tmp_dict)
                del args.cond["ribo"]
            # Conditions applied to transcript metadata
            for key, item in args.cond.items():
                # ignore if not parsing ribo-seq data
                if isinstance(item, dict) and args.use_ribo:
                    for id, cond in item.items():
                        # Update the corresponding condition for each matching group
                        for group_name, samples in args.ribo_paths.items():
                            paths = [s[1] for s in samples]
                            if id in paths:
                                tmp_dict = {f"{key}": lambda x: eval(cond)}
                                conds["grouped"][group_name].update(tmp_dict)
                else:
                    tmp_dict = {f"{key}": lambda x: eval(item)}
                    conds["global"].update(tmp_dict)

        args.cond = conds
        print(args)
        return args
