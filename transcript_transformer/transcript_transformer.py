import argparse
import sys
import os
import json
import yaml
import h5py
import numpy as np
import pandas as pd
from fasta_reader import read_fasta

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transcript_transformer.models import TranscriptSeqRiboEmb
from transcript_transformer.transcript_loader import h5pyDataModule, DNADatasetBatches, collate_fn
from torch.utils.data import DataLoader


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_json(args):
    with open(args.input_data, 'r') as fh:
        if args.input_data[-4:] == 'json':
            input_data = json.load(fh)
        else:
            input_data = yaml.safe_load(fh)
    args.__dict__.update(input_data)
    args.x_ribo = type(args.ribo) == dict
    if args.x_ribo is False:
        args.ribo = []
    # experimental
    if 'cond' in input_data.keys():
        args.cond = {k: eval(v) for k, v in input_data['cond'].items()}
    else:
        args.cond = None

    return args


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
        string += cdn_prot_dict[cdn]

    return string, has_stop


cdn_prot_dict = {
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


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass


class ParseArgs(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Transcript Transformer launch pad',
            usage='''transcript_transformer <command> [<args>]
             Commands:
               pretrain  Pretrain a model using self-supervised objective
               train     Train a model to detect TIS locations on transcripts
               predict   Predict TIS locations from input data
            ''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if args.command not in ['pretrain', 'train', 'predict']:
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        if args.command == 'pretrain':
            self.pretrain_train(mlm=True)
        elif args.command == 'train':
            self.pretrain_train(mlm=False)
        else:
            self.predict()

    def pretrain_train(self, mlm):
        parser = argparse.ArgumentParser(
            description=f'{"Pretrain transformer using MLM objective" if mlm else "train transcript transformer"}',
            formatter_class=CustomFormatter)
        # TWO argvs, ie the command (git) and the subcommand (commit)
        parser.add_argument('input_data', type=str, metavar='dict_path',
                            help="dictionary (json/yaml) path containing input data file info")
        parser.add_argument('--train', type=str, nargs='+',
                            help="contigs in data_path folder used for training. If not specified, "
                            "training is performed on all available contigs excluding val/test contigs")
        parser.add_argument('--val', type=str, nargs='+',
                            help="contigs in data_path folder used for validation")
        parser.add_argument('--test', type=str, nargs='+',
                            help="contigs in data_path folder used for testing")
        parser.add_argument('--ribo_offset', type=boolean, default=False,
                            help="offset mapped ribosome reads by read length")
        parser.add_argument('--name', type=str, default='',
                            help="name of the model")
        parser.add_argument('--log_dir', type=str, default='lightning_logs',
                            help="log dir")

        dl_parse = parser.add_argument_group(
            'DataLoader', 'Data loader arguments')
        dl_parse.add_argument('--min_seq_len', type=int, default=0,
                              help="minimum sequence length of transcripts")
        dl_parse.add_argument('--max_seq_len', type=int, default=30000,
                              help="maximum sequence length of transcripts")
        dl_parse.add_argument('--leaky_frac', type=float, default=0.05,
                              help="fraction of samples that escape conditions (ribo-seq)")
        dl_parse.add_argument('--num_workers', type=int, default=12,
                              help="number of data loader workers")
        dl_parse.add_argument('--max_memory', type=int, default=24000,
                              help="MB value applied for bucket batches based on rough estimates")
        dl_parse.add_argument('--max_transcripts_per_batch', type=int, default=2000,
                              help="maximum of transcripts per batch")

        tf_parse = parser.add_argument_group(
            'Model', f'Transformer arguments {"for MLM objective" if mlm else ""}')
        tf_parse.add_argument('--transfer_checkpoint', type=str,
                              help="Path to checkpoint pretrained model")
        tf_parse.add_argument('--lr', type=float, default=1e-3,
                              help="learning rate")
        tf_parse.add_argument('--decay_rate', type=float, default=0.96,
                              help="multiplicatively decays learning rate for every epoch")
        tf_parse.add_argument('--warmup_steps', type=int, default=1500,
                              help="number of warmup steps at the start of training")
        tf_parse.add_argument('--num_tokens', type=int, default=5,
                              help="number of unique nucleotide input tokens")
        tf_parse.add_argument('--dim', type=int, default=30,
                              help="dimension of the hidden states")
        tf_parse.add_argument('--depth', type=int, default=6,
                              help="number of layers")
        tf_parse.add_argument('--heads', type=int, default=6,
                              help="number of attention heads in every layer")
        tf_parse.add_argument('--dim_head', type=int, default=16,
                              help="dimension of the attention head matrices")
        tf_parse.add_argument('--nb_features', type=int, default=80,
                              help="number of random features, if not set, will default to (d * log(d)), "
                              "where d is the dimension of each head")
        tf_parse.add_argument('--feature_redraw_interval', type=int, default=1000,
                              help="how frequently to redraw the projection matrix")
        tf_parse.add_argument('--generalized_attention', type=boolean, default=True,
                              help="applies generalized attention functions")
        tf_parse.add_argument('--kernel_fn', type=boolean, default=torch.nn.ReLU(),
                              help="generalized attention function to apply (if generalized attention)")
        tf_parse.add_argument('--reversible', type=boolean, default=False,
                              help="reversible layers, from Reformer paper")
        tf_parse.add_argument('--ff_chunks', type=int, default=1,
                              help="chunk feedforward layer, from Reformer paper")
        tf_parse.add_argument('--use_scalenorm', type=boolean, default=False,
                              help="use scale norm, from 'Transformers without Tears' paper")
        tf_parse.add_argument('--use_rezero', type=boolean, default=False,
                              help="use rezero, from 'Rezero is all you need' paper")
        tf_parse.add_argument('--ff_glu', type=boolean, default=False,
                              help="use GLU variant for feedforward")
        tf_parse.add_argument('--emb_dropout', type=float, default=0.1,
                              help="embedding dropout")
        tf_parse.add_argument('--ff_dropout', type=float, default=0.1,
                              help="feedforward dropout")
        tf_parse.add_argument('--attn_dropout', type=float, default=0.1,
                              help="post-attn dropout")
        tf_parse.add_argument('--local_attn_heads', type=int, default=4,
                              help="the amount of heads used for local attention")
        tf_parse.add_argument('--local_window_size', type=int, default=256,
                              help="window size of local attention")
        tf_parse.add_argument('--debug', type=boolean, default=False,
                              help="debug mode disables logging and checkpointing (only for train)")
        tf_parse.add_argument('--patience', type=int, default=8,
                              help="Number of epochs required without the validation loss reducing"
                              "to stop training")
        tf_parse.add_argument('--mask_frac', type=float, default=0.85,
                              help="fraction of inputs that are masked, only for self-supervised training")
        tf_parse.add_argument('--rand_frac', type=float, default=0.10,
                              help="fraction of inputs that are randomized, only for self-supervised training")
        tf_parse.add_argument('--metrics', type=str, nargs='*', default=['ROC', 'PR'], choices=['ROC', 'PR'],
                              help="metrics calculated at the end of the epoch for the validation/test"
                              "set. These bring a cost to memory")

        tr_parse = parser.add_argument_group(
            'Trainer', 'Pytorch-lightning Trainer arguments')
        tr_parse.add_argument('--accelerator', type=str, default='cpu',
                              choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto'], help="computational hardware to apply")
        tr_parse.add_argument('--strategy', type=str, default='auto',
                              help="strategy for multi-gpu computation")
        tr_parse.add_argument('--devices', type=int,
                              default=0, nargs='+', help="device to use")
        tr_parse.add_argument('--max_epochs', type=int,
                              default=60, help="maximum epochs of training")

        args = parser.parse_args(sys.argv[2:])
        args = parse_json(args)
        if mlm:
            assert (not args.x_ribo) or (
                not args.seq), "only one type of data supported for self-supervised objective"
            assert not args.ribo_offset, "using a read length offset is not supported for self-supervised objective"
            args.mlm = 'seq' if args.seq else 'ribo'
        else:
            args.mlm = None
        args.num_tokens += 3  # add tokens used for data loading/padding
        print(f"{'Self-' if mlm else ''}Supervised learning:\n----------\n {args}")
        train(args)

    def predict(self):
        parser = argparse.ArgumentParser(description='Predict TIS locations',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('input_data', type=str, metavar='input_data',
                            help='path to json/yaml dict (h5) or fasta file, or RNA sequence')
        parser.add_argument('input_type', type=str, metavar='input_type',
                            help="type of input", choices=['h5', 'fa', 'RNA'])
        parser.add_argument('transfer_checkpoint', type=str, metavar='checkpoint',
                            help="path to checkpoint of trained model")
        parser.add_argument('--prob_th', type=float, default=0.01,
                            help="minimum prediction threshold at which additional information is processed")
        parser.add_argument('--save_path', type=str, metavar='save_path', default='results',
                            help="save file path")
        parser.add_argument('--output_type', type=str, default='npy', choices=['npy', 'h5'],
                            help="file type of raw model predictions")

        tr_parse = parser.add_argument_group(
            'Trainer', 'Pytorch-lightning Trainer arguments')
        tr_parse.add_argument('--accelerator', type=str, default='cpu',
                              choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto'], help="computational hardware to apply")
        tr_parse.add_argument('--strategy', type=str, default='auto',
                              help="strategy for multi-gpu computation")
        tr_parse.add_argument('--devices', type=str,
                              default='auto', nargs='+', help="device to use")

        dl_parse = parser.add_argument_group(
            'DataLoader', 'data loader arguments (when loading from h5 file)')
        dl_parse.add_argument('--test', type=str, nargs='+',
                              help="contigs to predict on (h5 input format only)")
        dl_parse.add_argument('--ribo_offset', type=boolean, default=False,
                              help="offset mapped ribosome reads by read length")
        dl_parse.add_argument('--min_seq_len', type=int, default=0,
                              help="minimum sequence length of transcripts")
        dl_parse.add_argument('--max_seq_len', type=int, default=30000,
                              help="maximum sequence length of transcripts")
        dl_parse.add_argument('--num_workers', type=int, default=12,
                              help="number of data loader workers")
        dl_parse.add_argument('--max_transcripts_per_batch', type=int, default=300,
                              help="maximum of transcripts per batch")
        dl_parse.add_argument('--max_memory', type=int, default=24000,
                              help="MB value applied for bucket batches based on rough estimates")
        dl_parse.add_argument('--metrics', type=str, nargs='*', default=['ROC', 'PR'], choices=['ROC', 'PR'],
                              help="metrics calculated at the end of the epoch for the validation/test"
                              "set. These bring a cost to memory")

        args = parser.parse_args(sys.argv[2:])

        print('Imputing labels from trained model: {}'.format(args))
        predict(args)


def train(args):
    if args.transfer_checkpoint:
        trans_model = TranscriptSeqRiboEmb.load_from_checkpoint(args.transfer_checkpoint, strict=False, x_seq=args.seq, x_ribo=args.x_ribo,
                                                                lr=args.lr, decay_rate=args.decay_rate, warmup_step=args.warmup_steps,
                                                                max_seq_len=args.max_seq_len, mlm=args.mlm, mask_frac=args.mask_frac,
                                                                rand_frac=args.rand_frac)
    else:
        trans_model = TranscriptSeqRiboEmb(args.seq, args.x_ribo, args.num_tokens, args.lr, args.decay_rate, args.warmup_steps,
                                           args.max_seq_len, args.dim, args.depth, args.heads, args.dim_head, False, args.nb_features,
                                           args.feature_redraw_interval, args.generalized_attention, args.kernel_fn,
                                           args.reversible, args.ff_chunks, args.use_scalenorm, args.use_rezero, False,
                                           args.ff_glu, args.emb_dropout, args.ff_dropout, args.attn_dropout,
                                           args.local_attn_heads, args.local_window_size, args.mlm, args.mask_frac, args.rand_frac, args.metrics)

    tr_loader = h5pyDataModule(args.h5_path, args.exp_path, args.ribo, args.y_path, args.seq, args.ribo_offset,
                               args.id_path, args.contig_path, train=args.train, val=args.val, test=args.test, max_memory=args.max_memory,
                               max_transcripts_per_batch=args.max_transcripts_per_batch, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len,
                               num_workers=args.num_workers, cond_fs=args.cond, leaky_frac=args.leaky_frac, collate_fn=collate_fn)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          filename="{epoch:02d}_{val_loss:.2f}", save_top_k=1, mode="min")
    tb_logger = pl.loggers.TensorBoardLogger(
        '.', os.path.join(args.log_dir, args.name))
    if args.debug:
        trainer = pl.Trainer(args.accelerator, args.strategy, args.devices, max_epochs=args.max_epochs, reload_dataloaders_every_n_epochs=1,
                             callbacks=[EarlyStopping(
                                 monitor="val_loss", mode="min", patience=args.patience)],
                             enable_checkpointing=False, logger=False)
    else:
        trainer = pl.Trainer(args.accelerator, args.strategy, args.devices, max_epochs=args.max_epochs, reload_dataloaders_every_n_epochs=1,
                             callbacks=[checkpoint_callback, EarlyStopping(
                                 monitor="val_loss", mode="min", patience=args.patience)],
                             logger=tb_logger)
    trainer.fit(trans_model, datamodule=tr_loader)
    trainer.test(trans_model, datamodule=tr_loader, ckpt_path='best')


def predict(args):
    assert args.input_type in [
        'h5', 'fa', 'RNA'], "input type not valid, must be one of 'h5', 'fa', or 'RNA'"
    trans_model = TranscriptSeqRiboEmb.load_from_checkpoint(args.transfer_checkpoint, strict=False, max_seq_len=args.max_seq_len,
                                                            mlm=False, mask_frac=0.85, rand_frac=0.15, metrics=[])
    trainer = pl.Trainer(args.accelerator, args.strategy,
                         args.devices, enable_checkpointing=False, logger=None)
    if args.input_type == 'h5':
        args = parse_json(args)
        tr_loader = h5pyDataModule(args.h5_path, args.exp_path, args.ribo, args.y_path, args.seq, args.ribo_offset,
                                   args.id_path, args.contig_path, test=args.test, max_memory=args.max_memory, max_transcripts_per_batch=args.max_transcripts_per_batch,
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
                tr_seqs.append(item.sequence)
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
        df.to_csv(f"{args.save_path}.csv")
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
            df.loc[num] = [ids[i], len(tr), idx+1, preds[i][idx], tr[idx:idx+3], TTS_pos, tr[TTS_pos:TTS_pos+3],
                           has_stop, len(prot_seq), prot_seq]
            num += 1
    return df


def main():
    args = ParseArgs()


if __name__ == "__main__":
    main()
