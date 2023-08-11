import itertools
import json
import yaml
import argparse
import numpy as np


class Parser(argparse.ArgumentParser):
    def __init__(self, stage="None", **kwargs):
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )
        if stage == "train":
            self.add_misc_args()
        if stage != "predict":
            self.add_argument(
                "input_config",
                type=str,
                help="config file (dict) containing input data info",
            )

    def add_data_args(self):
        input_parse = self.add_argument_group("Data processing arguments")
        input_parse.add_argument(
            "--overwrite",
            action="store_true",
            help="overwrite ribo-seq data when id is already present in h5 file",
        )
        input_parse.add_argument(
            "--no_backup",
            action="store_false",
            help="Do not create a backup of the processed assembly in the GTF folder location",
        )
        input_parse.add_argument(
            "--backup_path",
            type=str,
            default=None,
            help="path to backup location (defaults to GTF folder location)",
        )
        input_parse.add_argument(
            "--low_memory",
            action="store_true",
            help="polars setting that trades memory for perfomance. use on low-memory devices.",
        )
        input_parse.add_argument(
            "--cores",
            type=int,
            default=12,
            help="number of processor cores used for data processing",
        )

    def add_comp_args(self):
        comp_parse = self.add_argument_group("Computational resources arguments")
        comp_parse.add_argument(
            "--num_workers", type=int, default=8, help="number of processor cores"
        )
        comp_parse.add_argument(
            "--max_memory",
            type=int,
            default=24000,
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

    def add_architecture_args(self):
        tf_parse = self.add_argument_group("Model architecture arguments")
        tf_parse.add_argument(
            "--num_tokens",
            type=int,
            default=5,
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
            default=30000,
            help="maximum sequence length of transcripts",
        )
        dl_parse.add_argument(
            "--max_transcripts_per_batch",
            type=int,
            default=2000,
            help="maximum of transcripts per batch",
        )

    def add_train_loading_args(self, pretrain=False):
        dl_parse = self.add_argument_group("Data loading arguments")
        if not pretrain:
            dl_parse.add_argument(
                "--transfer_checkpoint",
                type=str,
                help="Path to checkpoint pretrained model",
            )
            dl_parse.add_argument(
                "--train",
                type=str,
                nargs="+",
                help="chromosomes used for training. If not specified, "
                "training is performed on all available chromosomes excluding val/test contigs",
            )
            dl_parse.add_argument(
                "--val",
                type=str,
                nargs="+",
                help="chromosomes used for validation",
            )
            dl_parse.add_argument(
                "--test",
                type=str,
                nargs="+",
                help="chromosomes used for testing",
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
        dl_parse.add_argument(
            "--ribo_offset",
            action="store_true",
            help="offset mapped ribosome reads by read length",
        )

    def add_training_args(self):
        tr_parse = self.add_argument_group("Data loading arguments")
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
            default=8,
            help="Number of epochs required without the validation loss reducing"
            "to stop training",
        )

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
            metavar="input_type",
            help="type of input",
            choices=["config", "fa", "RNA"],
        )

    def add_preds_args(self):
        pr_parse = self.add_argument_group("Model prediction processing arguments")
        pr_parse.add_argument(
            "--prob_th",
            type=float,
            default=0.01,
            help="minimum prediction threshold at which additional information is processed",
        )
        pr_parse.add_argument(
            "--out_prefix",
            type=str,
            default="results",
            help="path (prefix) of output files",
        )
        pr_parse.add_argument(
            "--output_type",
            type=str,
            default="npy",
            choices=["npy", "h5"],
            help="file type of raw model predictions",
        )

    def add_misc_args(self):
        ms_parse = self.add_argument_group("Miscellaneous arguments")
        ms_parse.add_argument(
            "--debug",
            action="store_true",
            help="debug mode disables logging and checkpointing (only for train)",
        )


def parse_config_file(args):
    # DEFAULT VALUES (overwritten if present)
    args.exp_path = "transcript"
    args.y_path = "tis"
    args.contig_path = "contig"
    args.id_path = "id"

    # read dict and add to args
    with open(args.input_config, "r") as fh:
        if args.input_config[-4:] == "json":
            input_config = json.load(fh)
        else:
            input_config = yaml.safe_load(fh)
    args.__dict__.update(input_config)

    # backward compatibility
    if "seq" in args:
        args.use_seq = args.seq
    cond_1 = (
        ("ribo_paths" in args)
        and (type(args.ribo_paths) == dict)
        and (len(args.ribo_paths) > 0)
    )
    cond_2 = ("ribo" in args) and (len(args.ribo) > 0)
    args.use_ribo = cond_1 or cond_2
    args.ribo_shifts = {}

    # ribo takes precedence over ribo_paths, can also includes read shifts
    if args.use_ribo:
        if cond_2:
            if type(args.ribo) == dict:
                args.ribo_ids = list(args.ribo.keys())
                args.ribo_shifts = args.ribo
            else:
                args.ribo_ids = args.ribo
        else:
            args.ribo_ids = list(args.ribo_paths.keys())
    else:
        args.ribo_ids = []

    # conditions used to remove transcripts from training data
    if "cond" in input_config.keys():
        args.cond = {k: eval(v) for k, v in input_config["cond"].items()}
    else:
        args.cond = None

    # creates sets of merged (or solo) datasets based on merged input
    if "merge" not in input_config.keys() or not args.use_ribo:
        args.merge = []
    args.merge_dict = {}
    merg_mask = ~np.isin(
        np.arange(len(args.ribo_ids)), list(itertools.chain(*args.merge))
    )
    for data_idx in np.where(merg_mask)[0]:
        args.merge += [[data_idx]]
    for i, set in enumerate(args.merge):
        args.merge_dict[i] = set

    return args
