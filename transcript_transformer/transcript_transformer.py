import argparse
import sys
import os
import numpy as np
import itertools
from fasta_reader import read_fasta

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .models import TranscriptSeqRiboEmb
from .transcript_loader import (
    h5pyDataModule,
    DNADatasetBatches,
    collate_fn,
)
from .util_functions import DNA2vec
from .processing import process_seq_preds
from .data import process_seq_data, process_ribo_data
from .argparser import Parser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcript Transformer launch pad",
        usage="""transcript_transformer <command> [<args>]
            Commands:
            data      process raw data for use with transcript-transformer
            pretrain  Pretrain a model using self-supervised objective
            train     Train a model to detect TIS locations on transcripts
            predict   Predict TIS locations from input data
        """,
    )
    parser.add_argument("command", help="Subcommand to run")
    args = parser.parse_args(sys.argv[1:2])
    if args.command not in ["data", "pretrain", "train", "predict"]:
        print("Unrecognized command")
        parser.print_help()
        exit(1)
    # use dispatch pattern to invoke method with same name
    if args.command == "data":
        parser = Parser(stage="data", description="Parse data in the h5 file")
        parser.add_data_args()
        args = parser.parse_arguments(sys.argv[2:])
        process_seq_data(
            args.h5_path, args.gtf_path, args.fa_path, args.backup_path, ~args.no_backup
        )
        if args.use_ribo:
            process_ribo_data(
                args.h5_path, args.ribo_paths, args.overwrite, args.low_memory
            )
    elif args.command == "pretrain":
        parser = Parser(
            stage="train", description="Pretrain transformer using MLM objective"
        )
        parser.add_train_loading_args()
        parser.add_selfsupervised_args()
        parser.add_training_args()
        parser.add_comp_args()
        parser.add_evaluation_args()
        parser.add_architecture_args()
        args = parser.parse_arguments(sys.argv[2:])
        assert not (
            args.use_ribo and args.use_seq
        ), "One input type allowed for self-supervised objective"
        assert args.offsets is None, "offsets not supported for MLM objective"
        args.mlm = "seq" if args.use_seq else "ribo"
        train(args)
    elif args.command == "train":
        parser = Parser(
            stage="train",
            description="Train a transformer using sequence or ribo-seq data",
        )
        parser.add_train_loading_args()
        parser.add_training_args()
        parser.add_comp_args()
        parser.add_evaluation_args()
        parser.add_architecture_args()
        args = parser.parse_arguments(sys.argv[2:])
        args.mlm, args.mask_frac, args.rand_frac = False, False, False
        train(args)
    else:
        parser = Parser(
            stage="predict", description="Predict translation initiation sites"
        )
        parser.add_custom_data_args()
        parser.add_predict_loading_args()
        parser.add_comp_args()
        parser.add_evaluation_args()
        parser.add_preds_args()
        args = parser.parse_arguments(sys.argv[2:])
        predict(args)


def train(args, test_model=True, enable_model_summary=True):
    if args.transfer_checkpoint:
        model = TranscriptSeqRiboEmb.load_from_checkpoint(
            args.transfer_checkpoint,
            strict=False,
            use_seq=args.use_seq,
            use_ribo=args.use_ribo,
            lr=args.lr,
            decay_rate=args.decay_rate,
            warmup_step=args.warmup_steps,
            max_seq_len=args.max_seq_len,
            mlm=args.mlm,
            mask_frac=args.mask_frac,
            rand_frac=args.rand_frac,
        )
    else:
        model = TranscriptSeqRiboEmb(
            args.use_seq,
            args.use_ribo,
            args.num_tokens,
            args.lr,
            args.decay_rate,
            args.warmup_steps,
            args.max_seq_len,
            args.dim,
            args.depth,
            args.heads,
            args.dim_head,
            False,
            args.nb_features,
            args.feature_redraw_interval,
            not args.no_generalized_attention,
            args.reversible,
            args.ff_chunks,
            args.use_scalenorm,
            args.use_rezero,
            False,
            args.ff_glu,
            args.emb_dropout,
            args.ff_dropout,
            args.attn_dropout,
            args.local_attn_heads,
            args.local_window_size,
            args.mlm,
            args.mask_frac,
            args.rand_frac,
            args.metrics,
        )
    tr_loader = h5pyDataModule(
        args.h5_path,
        args.exp_path,
        args.y_path,
        args.id_path,
        args.seqn_path,
        args.use_seq,
        args.ribo_ids,
        args.offsets,
        train=args.train,
        val=args.val,
        test=args.test,
        strict_validation=args.strict_validation,
        max_memory=args.max_memory,
        max_transcripts_per_batch=args.max_transcripts_per_batch,
        num_workers=args.num_workers,
        cond=args.cond,
        leaky_frac=args.leaky_frac,
        collate_fn=collate_fn,
        parallel=args.parallel,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:02d}_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    tb_logger = pl.loggers.TensorBoardLogger(".", os.path.join(args.log_dir, args.name))
    if args.debug:
        trainer = pl.Trainer(
            args.accelerator,
            args.strategy,
            args.devices,
            max_epochs=args.max_epochs,
            reload_dataloaders_every_n_epochs=1,
            enable_model_summary=enable_model_summary,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)
            ],
            enable_checkpointing=False,
            logger=False,
        )
    else:
        trainer = pl.Trainer(
            args.accelerator,
            args.strategy,
            args.devices,
            max_epochs=args.max_epochs,
            reload_dataloaders_every_n_epochs=1,
            enable_model_summary=enable_model_summary,
            callbacks=[
                checkpoint_callback,
                EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
            ],
            logger=tb_logger,
        )
    trainer.fit(model, datamodule=tr_loader)
    if test_model:
        print(trainer.checkpoint_callbacks)
        trainer.test(model, datamodule=tr_loader, ckpt_path="best")

    return trainer, model
    # trainer.predict(model, dataloaders=tr_loader, ckpt_path="best")


def predict(args, trainer=None, model=None, postprocess=True):
    if args.accelerator == "cpu":
        map_location = torch.device("cpu")
    else:
        map_location = torch.device("cuda")

    if trainer is None:
        trainer = pl.Trainer(
            args.accelerator,
            args.strategy,
            args.devices,
            enable_checkpointing=False,
            logger=None,
        )
    if model is None:
        model = TranscriptSeqRiboEmb.load_from_checkpoint(
            args.transfer_checkpoint,
            map_location=map_location,
            strict=False,
            max_seq_len=args.max_seq_len,
            mlm=False,
            mask_frac=0.85,
            rand_frac=0.15,
            metrics=[],
        )
        ckpt_path = None
    else:
        ckpt_path = "best"
    if args.input_type == "hdf5":
        tr_loader = h5pyDataModule(
            args.h5_path,
            args.exp_path,
            args.y_path,
            args.id_path,
            args.seqn_path,
            args.use_seq,
            args.ribo_ids,
            args.offsets,
            train=args.train,
            val=args.val,
            test=args.test,
            max_memory=args.max_memory,
            max_transcripts_per_batch=args.max_transcripts_per_batch,
            num_workers=args.num_workers,
            cond=args.cond,
            collate_fn=collate_fn,
            parallel=args.parallel,
        )
    else:
        if args.input_type == "RNA":
            tr_seqs = args.input_data.upper()
            x_data = [DNA2vec(tr_seqs)]
            tr_ids = ["seq_1"]
        elif args.input_type == "fasta":
            tr_ids = []
            tr_seqs = []
            for item in read_fasta(args.input_data):
                if len(item.sequence) < args.max_seq_len:
                    tr_ids.append(item.defline)
                    tr_seqs.append(item.sequence.upper())
                else:
                    f"Sequence {item.defline} is longer than {args.max_seq_len}, ommiting..."
            assert len(tr_seqs) > 0, "no valid sequences in fasta"
            x_data = [DNA2vec(seq) for seq in tr_seqs]
        tr_loader = DataLoader(
            DNADatasetBatches(tr_ids, x_data), collate_fn=collate_fn, batch_size=1
        )

    print("\nRunning sequences through model")
    out = trainer.predict(model, dataloaders=tr_loader, ckpt_path=ckpt_path)
    ids = list(itertools.chain(*[o[2] for o in out]))
    preds = list(itertools.chain(*[o[0] for o in out]))

    if args.input_type == "hdf5":
        targets = list(itertools.chain(*[o[1] for o in out]))
        out = [ids, preds, targets]
    else:
        out = [ids, preds]

    if postprocess:
        mask = [np.where(pred > args.min_prob)[0] for pred in preds]
        if len(np.hstack(mask)) > 0:
            df = process_seq_preds(ids, preds, tr_seqs, args.min_prob)
            print(df)
            df.to_csv(f"{args.out_prefix}.csv", index=None)
            print(f"\n--> Sites of interest saved to '{args.out_prefix}.csv'")
        else:
            print(
                f"\n!-> No sites of interest found (omitted creation of '{args.out_prefix}.csv')"
            )

    np.save(
        f"{args.out_prefix}.npy",
        np.array(out, dtype=object).T,
    )

    print(f"-->Raw model outputs saved to '{args.out_prefix}.npy'")

    return


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
