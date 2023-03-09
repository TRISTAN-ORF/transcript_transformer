import argparse
import sys
import os
import json
import h5py

from transcript_transformer.models import TranscriptMLM, TranscriptSeqRiboEmb
from transcript_transformer.transcript_loader import h5pyDataModule, collate_fn

import numpy as np
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass


class ParseArgs(object):
        def __init__(self):
            parser = argparse.ArgumentParser(
                        description='Transcript Transformer launch pad',
                        usage='''transcript_transformer <command> [<args>]
             Commands:
               pretrain  Pretrain a model using MLM objective
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
                                help="dictionary (json) path containing input data file info")
            parser.add_argument('--train', type=str, nargs='+',
                                help="contigs in data_path folder used for training. If not specified, "\
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

            dl_parse = parser.add_argument_group('DataLoader', 'data loader arguments')
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
            
            tf_parse = parser.add_argument_group('Model', f'Transformer arguments {"for MLM objective" if mlm else ""}')
            
            if mlm:
                tf_parse.add_argument('--mask_frac', type=float, default=0.85,
                                    help="fraction of inputs that are masked")
                tf_parse.add_argument('--rand_frac', type=float, default=0.10, 
                                    help="fraction of masked inputs that are randomized")
            else:
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
                                help="number of random features, if not set, will default to (d * log(d)), "\
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
            
            parser = pl.Trainer.add_argparse_args(parser)
            args = parser.parse_args(sys.argv[2:])
            args = parse_json(args)
            args.num_tokens += 3 # add tokens used for data loading/padding

            if mlm:
                print('Training a masked language model training with: {}'.format(args))
                mlm_train(args)
            else:
                print('Training a transcript transformer with: {}'.format(args))
                train(args)

        def predict(self):
            parser = argparse.ArgumentParser(description='Predict TIS locations',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser.add_argument('input_data', type=str, metavar='input_data',
                                help='path to JSON dict (h5) or fasta file, or RNA sequence')
            parser.add_argument('input_type', type=str, metavar='input_type',
                                help="type of input", choices=['h5', 'fa', 'RNA'])
            parser.add_argument('transfer_checkpoint', type=str, metavar='checkpoint',
                                help="path to checkpoint of trained model")
            parser.add_argument('--test', type=str, nargs='+',
                                help="contigs to predict on (h5 input format only)")
            parser.add_argument('--ribo_offset', type=boolean, default=False,
                                help="offset mapped ribosome reads by read length")
            parser.add_argument('--output_type', type=str, default='npy', choices=['npy', 'h5'],
                                help="file type of output predictions")
            parser.add_argument('--save_path', type=str, metavar='save_path', default='results',
                                help="save file path")
            dl_parse = parser.add_argument_group('DataLoader', 'data loader arguments')
            dl_parse.add_argument('--min_seq_len', type=int, default=0,
                                help="minimum sequence length of transcripts")
            dl_parse.add_argument('--max_seq_len', type=int, default=30000,
                                help="maximum sequence length of transcripts")
            dl_parse.add_argument('--num_workers', type=int, default=12, 
                                help="number of data loader workers")
            dl_parse.add_argument('--max_transcripts_per_batch', type=int, default=300, 
                                help="maximum of transcripts per batch")
            
            parser = pl.Trainer.add_argparse_args(parser)
            args = parser.parse_args(sys.argv[2:])
            
            print('Imputing labels from trained model: {}'.format(args))
            predict(args)

def parse_json(args):
    with open(args.input_data, 'r') as fh:
        input_data = json.load(fh)
        args.h5_path = input_data['h5_path']
        args.exp_path = input_data['exp_path']
        args.y_path = input_data['y_path']
        args.contig_path = input_data['contig_path']
        args.id_path = input_data['id_path']
        args.x_seq = input_data['seq']
        args.ribo_path = input_data['ribo']

        args.x_ribo = type(args.ribo_path) == dict
        if args.x_ribo is False:
            args.ribo_path = []
        # experimental
        if 'cond' in input_data.keys():
            args.cond = {k: eval(v) for k,v in input_data['cond'].items()}
        else:
            args.cond = None
    return args

def DNA2vec(dna_seq):
    seq_dict = {'A': 0, 'T': 1, 'U':1, 'C': 2, 'G': 3, 'N': 4}
    dna_vec = np.zeros(len(dna_seq), dtype=int)
    for idx in np.arange(len(dna_seq)):
        dna_vec[idx] = seq_dict[dna_seq[idx]]

    return dna_vec

def prep_input(x, device):
    x = torch.LongTensor(np.hstack(([5], x, [6]))).view(1,-1)
    y_mask = torch.isin(x, torch.Tensor([0,1,2,3,4]))
    
    return {'seq': x}, y_mask

def mlm_train(args):
    print(args.x_seq)
    mlm = TranscriptMLM(args.mask_frac, args.rand_frac, args.lr, args.decay_rate, args.warmup_steps, 
                        args.num_tokens, args.max_seq_len, args.dim, args.depth, args.heads, args.dim_head, 
                        False, args.nb_features, args.feature_redraw_interval, args.generalized_attention,
                        args.kernel_fn, args.reversible, args.ff_chunks, args.use_scalenorm, 
                        args.use_rezero, False, args.ff_glu, args.emb_dropout, args.ff_dropout,
                        args.attn_dropout, args.local_attn_heads, args.local_window_size)
    tr_loader = h5pyDataModule(args.h5_path, args.exp_path, args.ribo_path, args.y_path, args.x_seq, args.ribo_offset, 
                               args.id_path, args.contig_path, args.train, args.val, args.test, args.max_memory,
                               max_transcripts_per_batch=args.max_transcripts_per_batch, min_seq_len=args.min_seq_len, 
                               max_seq_len=args.max_seq_len, num_workers=args.num_workers, cond_fs=args.cond, collate_fn=collate_fn)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=os.path.join(args.log_dir, args.name), 
                                          filename="{epoch:02d}_{val_loss:.2f}", save_top_k=1, mode="min")
    tb_logger = pl.loggers.TensorBoardLogger('.', os.path.join(args.log_dir, args.name))
    trainer = pl.Trainer.from_argparse_args(args, reload_dataloaders_every_n_epochs=1, 
                                            callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=8)],
                                            logger=tb_logger)
    trainer.fit(mlm, datamodule=tr_loader)
    trainer.test(mlm, datamodule=tr_loader, ckpt_path='best')

def train(args):
    if args.transfer_checkpoint:
        trans_model = TranscriptSeqRiboEmb.load_from_checkpoint(args.transfer_checkpoint, strict=False, x_seq=args.x_seq, x_ribo=args.x_ribo,
                                                                lr=args.lr, decay_rate=args.decay_rate, warmup_step=args.warmup_steps, max_seq_len=args.max_seq_len)
    else:
        trans_model = TranscriptSeqRiboEmb(args.x_seq, args.x_ribo, args.num_tokens, args.lr, args.decay_rate, args.warmup_steps,
                args.max_seq_len, args.dim, args.depth, args.heads, args.dim_head, False, args.nb_features, 
                args.feature_redraw_interval, args.generalized_attention, args.kernel_fn, 
                args.reversible, args.ff_chunks, args.use_scalenorm, args.use_rezero, False,
                args.ff_glu, args.emb_dropout, args.ff_dropout, args.attn_dropout,
                args.local_attn_heads, args.local_window_size)
        
    tr_loader = h5pyDataModule(args.h5_path, args.exp_path, args.ribo_path, args.y_path, args.x_seq, args.ribo_offset, 
                               args.id_path, args.contig_path, train=args.train, val=args.val, test=args.test, max_memory=args.max_memory,
                               max_transcripts_per_batch=args.max_transcripts_per_batch, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, 
                               num_workers=args.num_workers, cond_fs=args.cond, leaky_frac=args.leaky_frac, collate_fn=collate_fn)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                          filename="{epoch:02d}_{val_loss:.2f}", save_top_k=1, mode="min")
    tb_logger = pl.loggers.TensorBoardLogger('.', os.path.join(args.log_dir, args.name))
    if args.debug:
        trainer = pl.Trainer.from_argparse_args(args, reload_dataloaders_every_n_epochs=1, enable_checkpointing=False, logger=False)
    else:
        trainer = pl.Trainer.from_argparse_args(args, reload_dataloaders_every_n_epochs=1,
                                                callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=8)],
                                                logger=tb_logger)
    trainer.fit(trans_model, datamodule=tr_loader)
    trainer.test(trans_model, datamodule=tr_loader, ckpt_path='best')

def predict(args):
    assert args.input_type in ['h5', 'fa', 'RNA'], "input type not valid, must be one of 'h5', 'fa', or 'RNA'"  
    device = torch.device('cuda') if args.gpus else torch.device('cpu')
    trans_model = TranscriptSeqRiboEmb.load_from_checkpoint(args.transfer_checkpoint)
    trans_model.to(device)
    trans_model.eval()
    out = {}

    if args.input_type == 'h5':
        args = parse_json(args)
        tr_loader = h5pyDataModule(args.h5_path, args.exp_path, args.ribo_path, args.y_path, args.x_seq, args.ribo_offset, 
                                args.id_path, args.contig_path, test=args.test, max_memory=args.max_memory, max_transcripts_per_batch=args.max_transcripts_per_batch,
                                min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, num_workers=args.num_workers, collate_fn=collate_fn)
        trainer = pl.Trainer(accelerator='gpu' if args.gpus else 'cpu', devices=1,  logger=False, enable_checkpointing=False)
        trainer.test(trans_model, dataloaders=tr_loader)
        out['id'] = np.array(trans_model.labels)
        out['pred'] = np.array([t.detach().cpu().numpy() for t in trans_model.test_outputs], dtype=object)
        out['target'] = np.array([t.detach().cpu().numpy() for t in trans_model.test_targets], dtype=object)
    else:
        if args.input_type == 'fa':
            file = open(args.input_data)
            data = file.readlines()
            file.close()
            tr_ids = data[0::2]
            tr_seqs = data[1::2]
            out['id'] = np.array([seq.replace('\n','') for seq in tr_ids])
            tr_seqs = [seq.replace('\n','').upper() for seq in tr_seqs]
            x_data = [DNA2vec(seq) for seq in tr_seqs if (len(seq) < args.max_seq_len) and (len(seq) > args.min_seq_len)]
            
        elif args.input_type == 'RNA':
            assert len(args.input_data) < args.max_seq_len, f'input is longer than maximum input length: {args.max_seq_len}'
            assert len(args.input_data) > args.min_seq_len, f'input is smaller than minimum input length: {args.min_seq_len}'
            x_data = [DNA2vec(args.input_data.upper())]
            out['id'] = np.array(['seq_1'])
        
        print('\nProcessing data')
        outputs = []
        for i,x in enumerate(x_data):
            print('\r{:.2f}%'.format(i/len(x_data)*100), end='')
            out = F.softmax(trans_model.forward(*prep_input(x, device)), dim=1)[:,1]
            outputs.append(out.detach().cpu().numpy())
        out['pred'] = np.array(outputs, dtype=object)
    
    if args.output_type == 'npy':
        np.save(args.save_path, np.array(list(out.values()), dtype=object).T)
    else:
        out_h = h5py.File(f'{args.save_path}', mode='w')
        grp = out_h.create_group('outputs')
        grp.create_dataset('id', data=out['id'].astype('S'),)
        dtype = h5py.vlen_dtype(np.dtype('float16')) if len(out['pred']) > 1 else np.float16
        grp.create_dataset('pred', data=out['pred'], dtype=dtype)
        if 'target' in out.keys():
            dtype = h5py.vlen_dtype(np.dtype('bool')) if len(out['pred']) > 1 else 'bool'
            grp.create_dataset('target', data=out['target'], dtype=dtype)
        out_h.close()
    print(f'\nResults saved to `{args.save_path}')

def main():
    args = ParseArgs()
    
if __name__ == "__main__":
    main()