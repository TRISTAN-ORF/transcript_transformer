#import h5pickle as h5py
import h5py
import numpy as np
import torch
from h5max import load_sparse_matrices
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def collate_fn(batch):
    """
    custom collate function used for adding the predermined tokens 5 and 6 to every transcript 
    sequence at the beginning and end. 
    in addition, the varying input length sequences are padded using the predetermined token 7.
    These tokens are only processed as nucleotide embeddings, if these are used by the model.
    """
    # In which cases is this true?
    if type(batch[0][0]) == list:
        batch = batch[0]
    lens = np.array([len(s) for s in batch[2]])
    max_len = max(lens)
    
    y_b = torch.LongTensor(np.array([np.pad(y,(1,1+l), constant_values=7) for y, l in zip(batch[2], max_len - lens)]))
    
    x_dict = {}
    for k in batch[1][0].keys():
        # if the entries are multidimensional: positions x read lengths (reads)
        if len(batch[1][0][k].shape) > 1:
            x_exp = [np.pad(x[k],((1,1),(0,0)), constant_values=((5,6),(0,0))) for x in batch[1]]
            x_exp = [np.pad(x,((0,l),(0,0)), constant_values=((0,7),(0,0))) for x, l in zip(x_exp, max_len - lens)]
            x_dict[k] = torch.FloatTensor(np.array(x_exp, dtype=float))
        
        # if the entries are single dimensional and float: positions (reads)
        elif batch[1][0][k].dtype == float:
            x_exp = [np.concatenate(([5], x[k], [6], [7]*l)) for x, l in zip(batch[1], max_len - lens)]
            x_dict[k] = torch.FloatTensor(np.array(x_exp, dtype=float)).unsqueeze(-1)
        
        # if the entries are single dimensional and string: positions (nucleotides)
        else:
            x_dict[k] = torch.LongTensor(np.array([np.concatenate(([5], x[k], [6], [7]*l)) for x, l in zip(batch[1], max_len - lens)], dtype=int))
            
    x_dict.update({'x_id':batch[0], 'y':y_b})
    
    return x_dict

def local_shuffle(data, lens=None):
    if lens is None:
        lens = np.array([ts[0].shape[0] for ts in data])
    elif type(lens) == list:
        lens = np.array(lens)
    # get split idxs representing spans of 400
    splits = np.arange(1,max(lens),400)
    # get idxs
    idxs = np.arange(len(lens))

    shuffled_idxs = []
    ### Local shuffle 
    for l, u in zip(splits, np.hstack((splits[1:],[999999]))):
        # mask between lower and upper
        mask = np.logical_and(l < lens, lens <= u)
        # get idxs within mask
        shuffled = idxs[mask]
        # randomly shuffle idxs
        np.random.shuffle(shuffled)
        # add to idxs all
        shuffled_idxs.append(shuffled)
    shuffled_idxs = np.hstack(shuffled_idxs)
    data = data[shuffled_idxs]
    lens = lens[shuffled_idxs]

    return data, lens

def bucket(data, lens, max_seq_len, max_transcripts_per_batch, min_seq_len=0):
    # split idx sites l
    l = []
    # idx pos
    num_samples = 0
    # filter invalid lens
    mask = np.logical_and(np.array(lens)<=max_seq_len, np.array(lens)>=min_seq_len)
    data = data[mask]
    lens = lens[mask]
    ### bucket batching
    while len(data) > num_samples:
        # get lens of leftover transcripts
        lens_set = lens[num_samples:]
        # calculate memory based on number and length of samples (+2 for transcript start/stop token)
        mask = (np.maximum.accumulate(lens_set)+2) * (np.arange(len(lens_set))+1) >= max_seq_len
        # obtain position where mem > max_memory
        mask_idx = np.where(mask)[0]
        # get idx to split
        if len(mask_idx) > 0 and (mask_idx[0] > 0):
            # max amount of transcripts per batch
            samples_d = min(mask_idx[0],max_transcripts_per_batch)
            num_samples += samples_d
            l.append(num_samples)       
        else:
            break
    # [:-1] not possible when trying to test all data
    return np.split(data, l)#[:-1]

class h5pyDataModule(pl.LightningDataModule):
    def __init__(self, h5py_path, exp_path, ribo_paths, y_path, x_seq=False, ribo_offset=False, id_path='id', contig_path='contig', 
                 val=[], test=[], max_transcripts_per_batch=500, min_seq_len=0, max_seq_len=30000, num_workers=5, 
                 cond_fs=None, leaky_frac=0.05, collate_fn=collate_fn):
        super().__init__()
        self.ribo_paths = ribo_paths
        self.ribo_offset = ribo_offset
        if ribo_offset:
            assert len(list(ribo_paths.values())) > 0, f"No offset values present in ribo_paths input, check the function docstring"
        # number of datasets
        self.n_data = max(len(self.ribo_paths), 1)
        self.x_seq = x_seq
        self.y_path = y_path
        self.id_path = id_path
        self.h5py_path = h5py_path
        self.exp_path = exp_path
        self.contig_path = contig_path
        self.val_contigs = np.ravel([val])
        self.test_contigs = np.ravel([test])
        self.max_transcripts_per_batch = max_transcripts_per_batch
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.num_workers = num_workers
        self.cond_fs = cond_fs
        self.leaky_frac = leaky_frac
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.fh = h5py.File(self.h5py_path,'r')[self.exp_path]
        self.cond_mask = np.full(len(self.fh[self.id_path]), True)
        
        if self.cond_fs is not None:
            for key, cond in self.cond_fs.items():
                self.cond_mask = np.logical_and(self.cond_mask, cond(np.array(self.fh[key])))
            if self.leaky_frac > 0:
                leaky_abs = int(np.sum(self.cond_mask)*self.leaky_frac)
                leaky_idxs = np.random.choice(np.where(~self.cond_mask)[0], leaky_abs)
                self.cond_mask[leaky_idxs] = True
        
        contigs = np.unique(self.fh[self.contig_path]).astype(str)
        for ct in self.val_contigs:
            contigs = np.delete(contigs, np.where(contigs == str(ct)))
        for ct in self.test_contigs:
            contigs = np.delete(contigs, np.where(contigs == str(ct)))
        self.train_contigs = contigs
        print(f"Training contigs: {self.train_contigs}")
        print(f"Validation contigs: {self.val_contigs}")
        print(f"Test contigs: {self.test_contigs}")
        
        if stage == "fit" or stage is None:
            contig_mask = np.isin(self.fh[self.contig_path], np.array(self.train_contigs).astype('S'))
            mask = np.logical_and(self.cond_mask, contig_mask)
            self.tr_idx, self.tr_len, self.tr_idx_adj = self.prepare_sets(mask)
            print(f"Training set transcripts: {len(self.tr_idx)}")
            mask = np.isin(self.fh[self.contig_path], self.val_contigs.astype('S'))
            self.val_idx, self.val_len, self.val_idx_adj = self.prepare_sets(mask)
            print(f"Validation set transcripts: {len(self.val_idx)}")
        if stage == "test" or stage is None:
            mask = np.isin(self.fh[self.contig_path], self.test_contigs.astype('S'))
            self.te_idx, self.te_len, self.te_idx_adj = self.prepare_sets(mask)
            print(f"Test set transcripts: {len(self.te_idx)}")
            
    def prepare_sets(self, mask):
        # idx mask
        idx_temp = np.where(mask)[0]
        # set idx shift value if multiple riboseq datasets are present
        set_idx_adj = np.max(idx_temp)+1
        set_idx = np.ravel([np.where(mask)[0]+(set_idx_adj*i) for i in np.arange(self.n_data)])
        set_len = list(self.fh['tr_len'][mask])*self.n_data
        
        return set_idx, set_len, set_idx_adj

    def train_dataloader(self):
        batches = bucket(*local_shuffle(self.tr_idx, self.tr_len), self.max_seq_len, self.max_transcripts_per_batch, self.min_seq_len)
        return DataLoader(h5pyDatasetBatches(self.fh, self.ribo_paths, self.y_path, self.id_path, self.x_seq, self.ribo_offset, self.tr_idx_adj, batches), 
                          collate_fn=collate_fn, num_workers=self.num_workers, shuffle=True, batch_size=1)

    def val_dataloader(self):
        batches = bucket(*local_shuffle(self.val_idx, self.val_len), self.max_seq_len, self.max_transcripts_per_batch, self.min_seq_len)
        return DataLoader(h5pyDatasetBatches(self.fh, self.ribo_paths, self.y_path, self.id_path, self.x_seq, self.ribo_offset, self.val_idx_adj, batches), 
                         collate_fn=self.collate_fn, num_workers=self.num_workers, batch_size=1)

    def test_dataloader(self):
        batches = bucket(*local_shuffle(self.te_idx, self.te_len), self.max_seq_len, self.max_transcripts_per_batch, self.min_seq_len)
        return DataLoader(h5pyDatasetBatches(self.fh, self.ribo_paths, self.y_path, self.id_path, self.x_seq, self.ribo_offset, self.te_idx_adj, batches),
                          collate_fn=self.collate_fn, num_workers=self.num_workers, batch_size=1)

class h5pyDatasetBatches(torch.utils.data.Dataset):
    def __init__(self, fh, ribo_paths, y_path, id_path, x_seq, ribo_offset, idx_adj, batches):
        super().__init__()
        self.fh = fh
        self.ribo_paths = ribo_paths
        self.y_path = y_path
        self.id_path = id_path
        self.x_seq = x_seq
        self.ribo_offset = ribo_offset
        self.idx_adj = idx_adj
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        # Transformation is performed when a sample is requested
        x_ids = []
        xs = []
        ys = []
        for idx_conc in self.batches[index]:
            idx = idx_conc % self.idx_adj
            # get transcript IDs 
            x_ids.append(self.fh[self.id_path][idx])
            x_dict = {}
            # get seq data
            if self.x_seq:
                x_dict['seq'] = self.fh['seq'][idx]
            # get ribo data
            if len(self.ribo_paths) > 0:
                # obtain data set and adjuster
                data_path = list(self.ribo_paths.keys())[idx_conc//self.idx_adj]
                x = load_sparse_matrices(self.fh[data_path], idx, format='csr')
                if self.ribo_offset:
                    col_names = np.array(self.fh[data_path]['col_names']).astype(str)
                    for col_key, shift in self.ribo_paths[data_path].items():
                        mask = col_names == col_key
                        if (shift != 0) and (shift > 0):
                            x[:shift, mask] = 0
                            x[shift:, mask] = x[:-shift, mask]
                        elif (shift != 0) and (shift < 0):
                            x[-shift:, mask] = 0
                            x[:-shift, mask] = x[shift:, mask]
                    x = x.sum(axis=1)
                    x = x/np.maximum(x.max(), 1)
                    x_dict['ribo_single'] = self.fh['seq'][idx]
                else:
                    x_dict['ribo_multi'] = x/np.maximum(np.sum(x, axis=1).max(), 1)
                
            xs.append(x_dict)
            ys.append(self.fh[self.y_path][idx])
            
        return [x_ids, xs, ys]