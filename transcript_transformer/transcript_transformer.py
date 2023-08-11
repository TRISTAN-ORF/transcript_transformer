import argparse
import sys
import os
import h5py
import numpy as np
import pandas as pd
from fasta_reader import read_fasta

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transcript_transformer.models import TranscriptSeqRiboEmb
from transcript_transformer.transcript_loader import h5pyDataModule, DNADatasetBatches, collate_fn
from transcript_transformer.data import process_data
from transcript_transformer.argparser import Parser, parse_config_file

CDN_PROT_DICT = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
    'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W'}


def DNA2vec(dna_seq):
    seq_dict = {'A': 0, 'T': 1, 'U': 1, 'C': 2, 'G': 3, 'N': 4}
    dna_vec = np.zeros(len(dna_seq), dtype=int)
    for idx in np.arange(len(dna_seq)):
        dna_vec[idx] = seq_dict[dna_seq[idx]]

    return dna_vec


def prep_input(x, device):
    x = torch.LongTensor(np.hstack(([5], x, [6]))).view(1, -1)
    y = torch.LongTensor(torch.ones_like(x))
    y[0, 0] = -1
    y[0, -1] = -1

    return {'seq': x, 'y': y}


def construct_prot(seq):
    stop_cds = ['TAG', 'TGA', 'TAA']
    sh_cds = np.array([seq[n:n+3] for n in range(0, len(seq)-2, 3)])
    stop_site_pos = np.where(np.isin(sh_cds, stop_cds))[0]
    if len(stop_site_pos) > 0:
        has_stop = True
        stop_site = stop_site_pos[0]
        cdn_seq = sh_cds[:stop_site]
    else:
        has_stop = False
        cdn_seq = sh_cds

    string = ''
    for cdn in cdn_seq:
        string += CDN_PROT_DICT[cdn]

    return string, has_stop


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transcript Transformer launch pad',
        usage='''transcript_transformer <command> [<args>]
            Commands:
            data      process raw data for use with transcript-transformer
            pretrain  Pretrain a model using self-supervised objective
            train     Train a model to detect TIS locations on transcripts
            predict   Predict TIS locations from input data
        ''')
    parser.add_argument('command', help='Subcommand to run')
    args = parser.parse_args(sys.argv[1:2])
    if args.command not in ['data', 'pretrain', 'train', 'predict']:
        print('Unrecognized command')
        parser.print_help()
        exit(1)
    # use dispatch pattern to invoke method with same name
    if args.command =='data':
        parser = Parser(stage='data', description="Parse data in the h5 file")
        parser.add_data_args()
        args = parser.parse_args(sys.argv[2:])
        args = parse_config_file(args)
        process_data(args)
    elif args.command == 'pretrain':
        parser = Parser(stage='train', description="Pretrain transformer using MLM objective")
        parser.add_train_loading_args()
        parser.add_selfsupervised_args()
        parser.add_training_args()
        parser.add_comp_args()
        parser.add_evaluation_args()
        parser.add_architecture_args()
        args = parser.parse_args(sys.argv[2:])
        args = parse_config_file(args)
        assert not (args.use_ribo and args.use_seq), "One input type allowed for self-supervised objective"
        assert not args.ribo_offset, "ribo_offset not supported for MLM objective"
        args.mlm = "seq" if args.use_seq else "ribo"
        train(args)
    elif args.command == 'train':
        parser = Parser(stage='train', description="Train a transformer using sequence or ribo-seq data")
        parser.add_train_loading_args()
        parser.add_training_args()
        parser.add_comp_args()
        parser.add_evaluation_args()
        parser.add_architecture_args()
        args = parser.parse_args(sys.argv[2:])
        args = parse_config_file(args)
        args.mlm, args.mask_frac, args.rand_frac = False, False, False
        train(args)
    else:
        parser = Parser(stage='predict', description="Predict translation initiation sites")
        parser.add_custom_data_args()
        parser.add_predict_loading_args()
        parser.add_comp_args()
        parser.add_evaluation_args()
        parser.add_preds_args()
        args = parser.parse_args(sys.argv[2:])
        if args.input_type == "config":
            args = parse_config_file(args)
        predict(args)

def train(args, predict=False, enable_model_summary=True):
    if args.transfer_checkpoint:
        trans_model = TranscriptSeqRiboEmb.load_from_checkpoint(args.transfer_checkpoint, strict=False, use_seq=args.use_seq, use_ribo=args.use_ribo,
                                                                lr=args.lr, decay_rate=args.decay_rate, warmup_step=args.warmup_steps,
                                                                max_seq_len=args.max_seq_len, mlm=args.mlm, mask_frac=args.mask_frac,
                                                                rand_frac=args.rand_frac)
    else:
        trans_model = TranscriptSeqRiboEmb(args.use_seq, args.use_ribo, args.num_tokens, args.lr, args.decay_rate, args.warmup_steps,
                                           args.max_seq_len, args.dim, args.depth, args.heads, args.dim_head, False, args.nb_features,
                                           args.feature_redraw_interval, not args.no_generalized_attention, torch.nn.ReLU(),
                                           args.reversible, args.ff_chunks, args.use_scalenorm, args.use_rezero, False,
                                           args.ff_glu, args.emb_dropout, args.ff_dropout, args.attn_dropout,
                                           args.local_attn_heads, args.local_window_size, args.mlm, args.mask_frac, args.rand_frac, args.metrics)

    tr_loader = h5pyDataModule(args.h5_path, args.exp_path, args.y_path, args.id_path, args.contig_path, args.use_seq, args.ribo_ids, args.ribo_shifts,
                               args.ribo_offset, args.merge_dict, train=args.train, val=args.val, test=args.test, max_memory=args.max_memory,
                               max_transcripts_per_batch=args.max_transcripts_per_batch, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len,
                               num_workers=args.num_workers, cond_fs=args.cond, leaky_frac=args.leaky_frac, collate_fn=collate_fn)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="{epoch:02d}_{val_loss:.2f}", save_top_k=1, mode="min")
    tb_logger = pl.loggers.TensorBoardLogger('.', os.path.join(args.log_dir, args.name))
    if args.debug:
        trainer = pl.Trainer(args.accelerator, args.strategy, args.devices, max_epochs=args.max_epochs, 
                             reload_dataloaders_every_n_epochs=1, enable_model_summary=enable_model_summary,
                             callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)],
                             enable_checkpointing=False, logger=False)
    else:
        trainer = pl.Trainer(args.accelerator, args.strategy, args.devices, max_epochs=args.max_epochs, reload_dataloaders_every_n_epochs=1,
                             enable_model_summary=enable_model_summary, 
                             callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)],
                             logger=tb_logger)
    trainer.fit(trans_model, datamodule=tr_loader)
    if not predict:
        print(trainer.checkpoint_callbacks)
        trainer.test(trans_model, datamodule=tr_loader, ckpt_path='best')
    else:
         return trainer.predict(trans_model, dataloaders=tr_loader, ckpt_path='best')

# TODO predict function needs refacturing and cleanup, better integration with custom riboformer scripts
def predict(args):
    if args.accelerator == 'cpu':
        map_location=torch.device('cpu')
    else:
        map_location=torch.device('cuda')
        
    trainer = pl.Trainer(args.accelerator, args.strategy,
                         args.devices, enable_checkpointing=False, logger=None)
    trans_model = TranscriptSeqRiboEmb.load_from_checkpoint(args.transfer_checkpoint, map_location=map_location, strict=False, max_seq_len=args.max_seq_len,
                                                            mlm=False, mask_frac=0.85, rand_frac=0.15, metrics=[])
    if args.input_type == 'config':
        tr_loader = h5pyDataModule(args.h5_path, args.exp_path, args.y_path, args.id_path, args.contig_path, args.use_seq, args.ribo_ids, args.ribo_shifts, 
                                   args.ribo_offset, args.merge_dict, test=args.test, max_memory=args.max_memory, max_transcripts_per_batch=args.max_transcripts_per_batch,
                                   min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        if args.input_type == 'RNA':
            tr_seqs = args.input_data.upper()
            x_data = [DNA2vec(tr_seqs)]
            tr_ids = 'seq_1'
        elif args.input_type == 'fa':
            tr_ids = []
            tr_seqs = []
            for item in read_fasta(args.input_data):
                tr_ids.append(item.defline)
                tr_seqs.append(item.sequence.upper())
            x_data = [DNA2vec(seq) for seq in tr_seqs]
        tr_loader = DataLoader(DNADatasetBatches(
            tr_ids, x_data), collate_fn=collate_fn, batch_size=1)

    print('\nRunning sequences through model')
    out = trainer.predict(trans_model, dataloaders=tr_loader)
    ids = np.array([o[2] for o in out])
    preds = np.array([o[0][0] for o in out], dtype=object)
    if args.input_type == 'h5':
        targets = np.array([o[1][0] for o in out], dtype=object)

    mask = [np.where(pred > args.prob_th)[0] for pred in preds]
    if len(np.hstack(mask)) > 0:
        df = process_results(mask, ids, preds, tr_seqs)
        print(df)
        df.to_csv(f"{args.save_path}.csv", index=None)
        print(f"\nSites of interest saved to '{args.save_path}.csv'")

    if args.output_type == 'npy':
        np.save(args.save_path, np.array(out, dtype=object))
    else:
        out_h = h5py.File(f'{args.save_path}', mode='w')
        grp = out_h.create_group('outputs')
        grp.create_dataset('id', data=ids.astype('S'),)
        dtype = h5py.vlen_dtype(np.dtype('float16')) if len(
            preds) > 1 else np.float16
        grp.create_dataset('pred', data=preds, dtype=dtype)
        if 'target' in out.keys():
            dtype = h5py.vlen_dtype(np.dtype('bool')) if len(
                preds) > 1 else 'bool'
            grp.create_dataset('target', data=targets, dtype=dtype)
        out_h.close()
    print(f"Raw model outputs saved to '{args.save_path}.{args.output_type}'")


def process_results(mask, ids, preds, seqs):
    df = pd.DataFrame(columns=['ID', 'tr_len', 'TIS_pos', 'output', 'start_codon', 'TTS_pos',
                               'TTS_codon', 'TTS_on_transcript', 'prot_len', 'prot_seq'])
    num = 0
    for i, idxs in enumerate(mask):
        tr = seqs[i]
        for idx in idxs:
            prot_seq, has_stop = construct_prot(tr[idx:])
            TTS_pos = idx+len(prot_seq)*3
            df.loc[num] = [ids[i][0],len(tr), idx+1, preds[i][idx], 
                           tr[idx:idx+3], TTS_pos, tr[TTS_pos:TTS_pos+3],
                           has_stop, len(prot_seq), prot_seq]
            num += 1
    return df


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
