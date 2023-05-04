import h5py
import numpy as np
import torch
from h5max import load_sparse
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

    y_b = torch.LongTensor(np.array(
        [np.pad(y, (1, 1+l), constant_values=-1) for y, l in zip(batch[2], max_len - lens)]))

    x_dict = {}
    for k in batch[1][0].keys():
        # if the entries are multidimensional: positions , read lengths
        if len(batch[1][0][k].shape) > 1:
            x_exp = [np.pad(x[k], ((1, 1), (0, 0)), constant_values=(
                (0, 0), (0, 0))) for x in batch[1]]
            x_exp = [np.pad(x.astype(float), ((0, l), (0, 0)), constant_values=(
                (0, 0.5), (0, 0))) for x, l in zip(x_exp, max_len - lens)]
            x_dict[k] = torch.FloatTensor(np.array(x_exp, dtype=float))

        # if the entries are single dimensional and float: positions (reads)
        elif batch[1][0][k].dtype == float:
            x_exp = [np.concatenate(([0], x[k], [0], [0.5]*l))
                     for x, l in zip(batch[1], max_len - lens)]
            x_dict[k] = torch.FloatTensor(
                np.array(x_exp, dtype=float)).unsqueeze(-1)

        # if the entries are single dimensional and string: positions (nucleotides)
        else:
            x_dict[k] = torch.LongTensor(np.array([np.concatenate(
                ([5], x[k], [6], [7]*l)) for x, l in zip(batch[1], max_len - lens)], dtype=int))

    x_dict.update({'x_id': batch[0], 'y': y_b})

    return x_dict


def local_shuffle(data, lens=None):
    if lens is None:
        lens = np.array([ts[0].shape[0] for ts in data])
    # get split idxs representing spans of 400
    splits = np.arange(1, max(lens), 400)
    # get idxs
    idxs = np.arange(len(lens))

    shuffled_idxs = []
    # Local shuffle
    for l, u in zip(splits, np.hstack((splits[1:], [999999]))):
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


def bucket(data, lens, max_memory, max_transcripts_per_batch, dataset):
    # split idx sites l
    l = []
    # idx pos
    num_samples = 0
    while len(data) > num_samples:
        # get lens of leftover transcripts
        lens_set = lens[num_samples:num_samples+max_transcripts_per_batch]
        # calculate memory based on number and length of samples (+2 for transcript start/stop token)
        num_samples_adj = np.multiply.accumulate(np.full(len(lens_set), 1.02))
        if dataset in ['train']:
            mask = (np.maximum.accumulate(lens_set+200)) * \
                (np.arange(len(lens_set))+1) * num_samples_adj < max_memory
        else:
            mask = (np.maximum.accumulate(lens_set+200)) * \
                (np.arange(len(lens_set))+1) * \
                num_samples_adj * 0.75 < max_memory
        # obtain position where mem > max_memory
        # get idx to split
        if sum(mask) > 0:
            samples_d = sum(mask)
            num_samples += samples_d
            l.append(num_samples)
        else:
            print(f"{len(data)-num_samples} ({(1-num_samples/len(data))*100:.2f}%) samples removed from {dataset} set because of memory constraints, "
                  "adjust max_memory to address behavior")
            data = data[:num_samples]
            lens = lens[:num_samples]

    assert len(data) > 0, f"No data samples left in {dataset} set"
    # always true?
    if l[-1] == len(data):
        l = l[:-1]
    return np.split(data, l)


class h5pyDataModule(pl.LightningDataModule):
    def __init__(self, h5py_path, exp_path, ribo_paths, y_path, x_seq=False, ribo_offset=False, id_path='id', contig_path='contig',
                 train=[], val=[], test=[], max_memory=24000, max_transcripts_per_batch=500, min_seq_len=0, max_seq_len=30000, num_workers=5,
                 cond_fs=None, leaky_frac=0.05, collate_fn=collate_fn):
        super().__init__()
        self.ribo_paths = ribo_paths
        self.ribo_offset = ribo_offset
        if ribo_offset:
            assert len(list(ribo_paths.values(
            ))) > 0, f"No offset values present in ribo_paths input, check the function docstring"
        # support for training on multiple datasets
        self.n_data = max(len(self.ribo_paths), 1)
        self.x_seq = x_seq
        self.y_path = y_path
        self.id_path = id_path
        self.h5py_path = h5py_path
        self.exp_path = exp_path
        self.contig_path = contig_path
        self.train_contigs = np.ravel([train]) if train else []
        self.val_contigs = np.ravel([val]) if val else []
        self.test_contigs = np.ravel([test]) if test else []
        self.max_memory = max_memory
        self.max_transcripts_per_batch = max_transcripts_per_batch
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.num_workers = num_workers
        self.cond_fs = cond_fs
        self.leaky_frac = leaky_frac
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.fh = h5py.File(self.h5py_path, 'r')[self.exp_path]
        # filter data
        tr_lens = np.array(self.fh['tr_len'])
        self.seq_len_mask = np.logical_and(
            tr_lens < self.max_seq_len, tr_lens > self.min_seq_len)
        
        # custom conditions # needs a rewrite
        self.global_cond_mask = np.full_like(self.seq_len_mask, True)
        if len(self.ribo_paths) > 0:
            cond = {key: np.full_like(self.seq_len_mask, True) for key in self.ribo_paths.keys()}
        else:
            cond = {'seq': np.full_like(self.seq_len_mask, True)}
        # Optional: implement custom masking for eval step
        cond_eval = cond.copy()
        if self.cond_fs is not None:
            ribo_path_keys = list(self.ribo_paths.keys())
            for key, cond in self.cond_fs.items():
                is_cond_ribo = np.core.defchararray.find(
                    key, ribo_path_keys) != -1
                temp_mask = cond(np.array(self.fh[key]))
                if self.leaky_frac > 0:
                    leaky_mask = np.random.uniform(
                        size=self.seq_len_mask.shape) > (1-self.leaky_frac)
                    temp_mask[leaky_mask] = True
                # apply mask to specific ribo data
                if is_cond_ribo.any():
                    cond[np.array(ribo_path_keys)[
                        is_cond_ribo][0]] = temp_mask
                # apply mask to all data
                else:
                    self.global_cond_mask = np.logical_and(
                        self.global_cond_mask, temp_mask)

        if (len(self.train_contigs) == 0) and ((stage == "fit") or (stage is None)):
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
            contig_mask = np.isin(self.fh[self.contig_path], np.array(
                self.train_contigs).astype('S'))
            mask = np.logical_and.reduce(
                [self.global_cond_mask, contig_mask, self.seq_len_mask])
            self.tr_idx, self.tr_len, self.tr_idx_adj = self.prepare_sets(
                mask, cond)
            print(f"Training set transcripts: {len(self.tr_idx)}")
            contig_mask = np.isin(self.fh[self.contig_path],
                                  self.val_contigs.astype('S'))
            self.val_idx, self.val_len, self.val_idx_adj = self.prepare_sets(
                np.logical_and(contig_mask, self.seq_len_mask), cond_eval)
            print(f"Validation set transcripts: {len(self.val_idx)}")
        if stage in ["test", "predict"] or stage is None:
            contig_mask = np.isin(self.fh[self.contig_path],
                                  self.test_contigs.astype('S'))
            self.te_idx, self.te_len, self.te_idx_adj = self.prepare_sets(
                np.logical_and(contig_mask, self.seq_len_mask), cond_eval)
            print(f"Test set transcripts: {len(self.te_idx)}")

    def prepare_sets(self, mask, cond):
        mask_set = []
        len_set = []
        sample_count = []
        tr_len = np.array(self.fh['tr_len'])
        for _, cond_mask in cond.items():
            mask_set.append(np.logical_and(cond_mask, mask))
            sample_count.append(sum(mask_set[-1]))
            len_set.append(tr_len[mask_set[-1]])
        mask = np.hstack(mask_set)
        lens = np.hstack(len_set)
        idxs = np.where(mask)[0]
        sort_idxs = np.argsort(lens)

        return idxs[sort_idxs], lens[sort_idxs], len(mask)

    def train_dataloader(self):
        batches = bucket(*local_shuffle(self.tr_idx, self.tr_len),
                         self.max_memory, self.max_transcripts_per_batch, 'train')
        return DataLoader(h5pyDatasetBatches(self.fh, self.ribo_paths, self.y_path, self.id_path, self.x_seq, self.ribo_offset, self.tr_idx_adj, batches),
                          collate_fn=collate_fn, num_workers=self.num_workers, shuffle=True, batch_size=1)

    def val_dataloader(self):
        batches = bucket(self.val_idx, self.val_len,
                         self.max_memory, self.max_transcripts_per_batch, 'val')
        return DataLoader(h5pyDatasetBatches(self.fh, self.ribo_paths, self.y_path, self.id_path, self.x_seq, self.ribo_offset, self.val_idx_adj, batches),
                          collate_fn=self.collate_fn, num_workers=self.num_workers, batch_size=1)

    def test_dataloader(self):
        batches = bucket(self.te_idx, self.te_len, self.max_memory,
                         self.max_transcripts_per_batch, 'test')
        return DataLoader(h5pyDatasetBatches(self.fh, self.ribo_paths, self.y_path, self.id_path, self.x_seq, self.ribo_offset, self.te_idx_adj, batches),
                          collate_fn=self.collate_fn, num_workers=self.num_workers, batch_size=1)

    def predict_dataloader(self):
        batches = bucket(self.te_idx, self.te_len, self.max_memory,
                         self.max_transcripts_per_batch, 'test')
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
            # get adjusted idx if multiple datasets are used
            idx = int(idx_conc % self.idx_adj)
            # get transcript IDs
            x_ids.append(self.fh[self.id_path][idx])
            x_dict = {}
            # get seq data
            if self.x_seq:
                x_dict['seq'] = self.fh[self.x_seq][idx]
            # get ribo data
            if len(self.ribo_paths) > 0:
                # obtain data set and adjuster
                data_path = list(self.ribo_paths.keys())[
                    idx_conc//self.idx_adj]
                x = load_sparse(self.fh[data_path], idx, format='csr').T
                if self.ribo_offset:
                    #col_names = np.array(self.fh[data_path]['col_names']).astype(str)
                    for col_i, (col_key, shift) in enumerate(self.ribo_paths[data_path].items()):
                        #mask = col_names == col_key
                        if (shift != 0) and (shift > 0):
                            x[:shift, col_i] = 0
                            x[shift:, col_i] = x[:-shift, col_i]
                        elif (shift != 0) and (shift < 0):
                            x[-shift:, col_i] = 0
                            x[:-shift, col_i] = x[shift:, col_i]
                    # get total number of reads per position
                    x = x.sum(axis=1)
                    # normalize including all positions,reads
                    x_dict['ribo'] = x/np.maximum(x.max(), 1)
                else:
                    # normalize including all positions,reads
                    x_dict['ribo'] = x/np.maximum(np.sum(x, axis=1).max(), 1)

            xs.append(x_dict)
            ys.append(self.fh[self.y_path][idx])

        return [x_ids, xs, ys]


class DNADatasetBatches(torch.utils.data.Dataset):
    def __init__(self, ids, xs):
        super().__init__()
        self.ids = ids
        self.xs = xs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Transformation is performed when a sample is requested
        x_ids = [self.ids[index]]
        xs = [{'seq': self.xs[index]}]
        ys = [np.ones_like(self.xs[index])]

        return [x_ids, xs, ys]
