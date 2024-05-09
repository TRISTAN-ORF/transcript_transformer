import h5py
import numpy as np
import torch
from h5max import load_sparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def collate_fn(batch):
    """
    custom collate function used for adding the predermined tokens 5 and 6 to every transcript
    sequence at the beginning and end.
    in addition, the varying input length sequences are padded using the predetermined token 7.
    These tokens are only processed as nucleotide embeddings, if these are used by the model.
    """
    if type(batch[0][0]) == list:
        batch = batch[0]
    lens = np.array([len(s) for s in batch[2]])
    max_len = max(lens)

    y_b = torch.LongTensor(
        np.array(
            [
                np.pad(y, (1, 1 + l), constant_values=-1)
                for y, l in zip(batch[2], max_len - lens)
            ]
        )
    )
    x_dict = {}
    for k in batch[1][0].keys():
        # if the entries are multidimensional: positions , read lengths
        if len(batch[1][0][k].shape) > 1:
            x_exp = [
                np.pad(x[k], ((1, 1), (0, 0)), constant_values=((0, 0), (0, 0)))
                for x in batch[1]
            ]
            x_exp = [
                np.pad(
                    x.astype(float),
                    ((0, l), (0, 0)),
                    constant_values=((0, 0.5), (0, 0)),
                )
                for x, l in zip(x_exp, max_len - lens)
            ]
            x_dict[k] = torch.FloatTensor(np.array(x_exp, dtype=float))

        # if the entries are single dimensional and float: positions (reads)
        elif batch[1][0][k].dtype == float:
            x_exp = [
                np.concatenate(([0], x[k], [0], [0.5] * l))
                for x, l in zip(batch[1], max_len - lens)
            ]
            x_dict[k] = torch.FloatTensor(np.array(x_exp, dtype=float)).unsqueeze(-1)

        # if the entries are single dimensional and string: positions (nucleotides)
        else:
            arr = np.empty(shape=(len(batch[1]), max_len + 2), dtype=int)
            for i, (x, l) in enumerate(zip(batch[1], max_len - lens)):
                arr[i] = np.concatenate(([5], x[k], [6], [7] * l))
            x_dict[k] = torch.LongTensor(arr)

    x_dict.update({"x_id": batch[0], "y": y_b})

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
        lens_set = lens[num_samples : num_samples + max_transcripts_per_batch]
        # calculate memory based on number and length of samples (+2 for transcript start/stop token)
        num_samples_adj = np.multiply.accumulate(np.full(len(lens_set), 1.02))
        if dataset in ["train"]:
            mask = (np.maximum.accumulate(lens_set + 200)) * (
                np.arange(len(lens_set)) + 1
            ) * num_samples_adj < max_memory
        else:
            mask = (np.maximum.accumulate(lens_set + 200)) * (
                np.arange(len(lens_set)) + 1
            ) * num_samples_adj * 0.75 < max_memory
        # obtain position where mem > max_memory
        # get idx to split
        if sum(mask) > 0:
            samples_d = sum(mask)
            num_samples += samples_d
            l.append(num_samples)
        else:
            print(
                f"{len(data)-num_samples} ({(1-num_samples/len(data))*100:.2f}%) samples removed from {dataset} set because of memory constraints, "
                "adjust max_memory to address behavior"
            )
            data = data[:num_samples]
            lens = lens[:num_samples]

    assert len(data) > 0, f"No data samples left in {dataset} set"
    # always true?
    if l[-1] == len(data):
        l = l[:-1]
    return np.split(data, l)


class h5pyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path,
        exp_path,
        y_path,
        tr_id_path,
        seqn_path,
        use_seq,
        ribo_ids,
        offsets,
        train=[],
        val=[],
        test=[],
        strict_validation=False,
        max_memory=24000,
        max_transcripts_per_batch=500,
        num_workers=5,
        cond=None,
        leaky_frac=0.05,
        collate_fn=collate_fn,
        parallel=False,
    ):
        super().__init__()
        self.ribo_ids = ribo_ids
        self.offsets = offsets
        # support for training on multiple datasets
        self.n_data = max(len(self.ribo_ids), 1)
        self.use_seq = use_seq
        self.y_path = y_path
        self.tr_id_path = tr_id_path
        self.h5_path = h5_path
        self.exp_path = exp_path
        self.seqn_path = seqn_path
        train = [t.encode("ascii") if type(t) is str else t for t in train]
        val = [t.encode("ascii") if type(t) is str else t for t in val]
        test = [t.encode("ascii") if type(t) is str else t for t in test]
        self.seqns = {
            "train": np.array(train),
            "val": np.array(val),
            "test": np.array(test),
        }
        self.strict_validation = strict_validation
        self.max_memory = max_memory
        self.max_transcripts_per_batch = max_transcripts_per_batch
        self.num_workers = num_workers
        if cond == None:
            self.cond = {"global": {}, "grouped": [{}] * len(ribo_ids)}
        else:
            self.cond = cond
        self.leaky_frac = leaky_frac
        self.collate_fn = collate_fn
        self.parallel = parallel

    def setup(self, stage=None):
        f = h5py.File(self.h5_path, "r")[self.exp_path]
        self.seqn_list = np.array(f[self.seqn_path])
        self.transcript_lens = np.array(f["tr_len"])
        # evaluate conditions
        # Identical mask over the samples applied to all datasets
        global_masks = []
        for key, cond_f in self.cond["global"].items():
            mask = cond_f(np.array(f[key]))
            if (key != "tr_len") and (self.leaky_frac > 0):
                prob_mask = np.random.uniform(size=len(mask)) > (1 - self.leaky_frac)
                mask[prob_mask] = True
            global_masks.append(mask)
        global_mask = np.logical_and.reduce(global_masks)
        # Masks over the samples applied per group of datasets
        group_masks = []
        for group in self.cond["grouped"]:
            group_mask = [np.full_like(global_mask, True)]
            for grp_idx, (key, cond_f) in enumerate(group.items()):
                # collect read counts for all samples in group
                if self.parallel:
                    read_data = []
                    for ribo_id in self.ribo_ids[grp_idx]:
                        h5_ribo_path = f"{self.h5_path.split('.h5')[0]}_{ribo_id}.h5"
                        r = h5py.File(h5_ribo_path, "r")[self.exp_path]
                        read_data.append(np.array(r[f"riboseq/{ribo_id}/5/{key}"]))
                        r.file.close()
                else:
                    read_data = [
                        np.array(f[f"riboseq/{ribo_id}/5/{key}"])
                        for ribo_id in self.ribo_ids[grp_idx]
                    ]
                grouped_feature = np.add.reduce(read_data)
                # apply function given in config file
                mask = cond_f(grouped_feature)
                # if data filtering is not based on transcript length,
                # allow leaky filtering
                if (key != "tr_len") and (self.leaky_frac > 0):
                    prob_mask = np.random.uniform(size=len(mask)) > (
                        1 - self.leaky_frac
                    )
                    mask[prob_mask] = True
                group_mask.append(mask)
            group_masks.append(np.logical_and.reduce(group_mask))
        f.file.close()
        # derive train/validation/test sets
        has_train_seqns = len(self.seqns["train"]) > 0
        has_val_seqns = len(self.seqns["val"]) > 0
        has_test_seqns = len(self.seqns["test"]) > 0
        filled_sets = sum([has_train_seqns, has_val_seqns, has_test_seqns])
        if filled_sets == 2:
            seqns = np.unique(self.seqn_list)
            if len(self.seqns["train"]) == 0:
                self.seqns["train"] = np.setdiff1d(
                    seqns, np.hstack((self.seqns["val"], self.seqns["test"]))
                )
            elif len(self.seqns["val"]) == 0:
                self.seqns["val"] = np.setdiff1d(
                    seqns, np.hstack((self.seqns["train"], self.seqns["test"]))
                )
            else:
                self.seqns["test"] = np.setdiff1d(
                    seqns, np.hstack((self.seqns["train"], self.seqns["val"]))
                )

        if stage == "fit" or stage is None:
            print(f"--> Training seqnames: {[t.decode() for t in self.seqns['train']]}")
            print(f"--> Validation seqnames: {[t.decode() for t in self.seqns['val']]}")
            if len(self.seqns["test"]) > 0:
                print(f"--> Test seqnames: {[t.decode() for t in self.seqns['test']]}")
            # train set
            seqn_mask = np.isin(self.seqn_list, np.array(self.seqns["train"]))
            mask = np.logical_and(global_mask, seqn_mask)
            self.tr_idx, self.tr_len, self.tr_idx_adj = self.prepare_sets(
                mask, group_masks
            )
            print(f"--> Training set transcripts: {len(self.tr_idx)}")
            assert len(self.tr_idx) > 0, "No transcripts in training data"
            # validation set
            seqn_mask = np.isin(self.seqn_list, self.seqns["val"])
            if self.strict_validation:
                # global_masks[0] is transcript length maskq
                mask = np.logical_and(seqn_mask, global_masks[0])
                self.val_idx, self.val_len, self.val_idx_adj = self.prepare_sets(
                    mask, [np.full_like(mask, True)] * len(group_masks)
                )
            else:
                mask = np.logical_and(global_mask, seqn_mask)
                self.val_idx, self.val_len, self.val_idx_adj = self.prepare_sets(
                    mask, group_masks
                )
            print(f"--> Validation set transcripts: {len(self.val_idx)}")
            assert len(self.val_idx) > 0, "No transcripts in validation data"
        if stage in ["test", "predict"] or stage is None:
            print(f"--> Test seqnames: {[t.decode() for t in self.seqns['test']]}")
            seqn_mask = np.isin(self.seqn_list, self.seqns["test"])
            # mask = np.logical_and(seqn_mask, global_masks[0])
            mask = np.logical_and(seqn_mask, global_masks[0])
            self.te_idx, self.te_len, self.te_idx_adj = self.prepare_sets(
                mask,
                [np.full_like(seqn_mask, True)] * len(group_masks),
            )
            print(f"--> Test set transcripts: {len(self.te_idx)}")
            assert len(self.te_idx) > 0, "No transcripts in test data"

    def prepare_sets(self, mask, group_conds):
        mask_set = []
        len_set = []
        for cond_mask in group_conds:
            mask_set.append(np.logical_and(cond_mask, mask))
            len_set.append(self.transcript_lens[mask_set[-1]])
        mask_all = np.hstack(mask_set)
        lens = np.hstack(len_set)
        idxs = np.where(mask_all)[0]
        sort_idxs = np.argsort(lens)

        return idxs[sort_idxs], lens[sort_idxs], len(mask)

    def train_dataloader(self):
        batches = bucket(
            *local_shuffle(self.tr_idx, self.tr_len),
            self.max_memory,
            self.max_transcripts_per_batch,
            "train",
        )
        return DataLoader(
            h5pyDatasetBatches(
                self.h5_path,
                self.y_path,
                self.tr_id_path,
                self.use_seq,
                self.ribo_ids,
                self.offsets,
                self.tr_idx_adj,
                batches,
                self.parallel,
            ),
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=1,
        )

    def val_dataloader(self):
        batches = bucket(
            self.val_idx,
            self.val_len,
            self.max_memory,
            self.max_transcripts_per_batch,
            "val",
        )
        return DataLoader(
            h5pyDatasetBatches(
                self.h5_path,
                self.y_path,
                self.tr_id_path,
                self.use_seq,
                self.ribo_ids,
                self.offsets,
                self.val_idx_adj,
                batches,
                self.parallel,
            ),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            batch_size=1,
        )

    def test_dataloader(self):
        batches = bucket(
            self.te_idx,
            self.te_len,
            self.max_memory,
            self.max_transcripts_per_batch,
            "test",
        )
        return DataLoader(
            h5pyDatasetBatches(
                self.h5_path,
                self.y_path,
                self.tr_id_path,
                self.use_seq,
                self.ribo_ids,
                self.offsets,
                self.te_idx_adj,
                batches,
                self.parallel,
            ),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            batch_size=1,
        )

    def predict_dataloader(self):
        batches = bucket(
            self.te_idx,
            self.te_len,
            self.max_memory,
            self.max_transcripts_per_batch,
            "test",
        )
        return DataLoader(
            h5pyDatasetBatches(
                self.h5_path,
                self.y_path,
                self.tr_id_path,
                self.use_seq,
                self.ribo_ids,
                self.offsets,
                self.te_idx_adj,
                batches,
                self.parallel,
            ),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            batch_size=1,
        )


class h5pyDatasetBatches(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path,
        y_path,
        tr_id_path,
        use_seq,
        ribo_ids,
        offsets,
        idx_adj,
        batches,
        parallel,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.ribo_ids = ribo_ids
        self.y_path = y_path
        self.tr_id_path = tr_id_path
        self.use_seq = use_seq
        self.offsets = offsets
        self.idx_adj = idx_adj
        self.batches = batches
        self.parallel = parallel

    def __len__(self):
        return len(self.batches)

    def open_hdf5(self):
        if self.parallel:
            h5_base = self.h5_path.split(".h5")[0]
            # flatten nested list
            flat_ids = sum(self.ribo_ids, [])
            self.r = {
                id: h5py.File(f"{h5_base}_{id}.h5", "r")["transcript"]
                for id in flat_ids
            }
        self.f = h5py.File(self.h5_path, "r")["transcript"]

    def __getitem__(self, index):
        if not hasattr(self, "fh"):
            self.open_hdf5()
        # Transformation is performed when a sample is requested
        x_ids = []
        xs = []
        ys = []
        for idx_conc in self.batches[index]:
            # get adjusted idx if multiple datasets are used
            idx = int(idx_conc % self.idx_adj)
            x_dict = {}
            # get seq data
            if self.use_seq:
                x_dict["seq"] = self.f["seq"][idx]
                id_prefix = ""
            # get ribo data
            else:
                # determine data set from grouped (concatenated) datasets
                set_idx = idx_conc // self.idx_adj
                x_merge = []
                ribo_set = self.ribo_ids[set_idx]
                # iterate ribo-seq experiments in merged (summed) datasets
                for i, ribo_id in enumerate(ribo_set):
                    if i == 0:
                        id_prefix = "&".join(ribo_set) + "|"
                    ribo_path = f"riboseq/{ribo_id}/5"
                    if self.parallel:
                        x = load_sparse(self.r[ribo_id][ribo_path], idx, format="csr").T
                    else:
                        x = load_sparse(self.f[ribo_path], idx, format="csr").T
                    if self.offsets is not None:
                        for col_i, (_, shift) in enumerate(
                            self.offsets[ribo_id].items()
                        ):
                            if (shift != 0) and (shift > 0):
                                x[:shift, col_i] = 0
                                x[shift:, col_i] = x[:-shift, col_i]
                            elif (shift != 0) and (shift < 0):
                                x[-shift:, col_i] = 0
                                x[:-shift, col_i] = x[shift:, col_i]
                        # get total number of reads per position
                        x = x.sum(axis=1)
                    x_merge.append(x)
                # sum merged data sets
                x = np.sum(x_merge, axis=0)
                if self.offsets is not None:
                    # normalize including all positions,reads
                    x_dict["ribo"] = x / np.maximum(x.max(), 1)
                else:
                    # normalize including all positions,reads
                    x_dict["ribo"] = x / np.maximum(np.sum(x, axis=1).max(), 1)
            # get transcript IDs
            x_ids.append(id_prefix.encode() + self.f[self.tr_id_path][idx])
            xs.append(x_dict)
            ys.append(self.f[self.y_path][idx])
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
        xs = [{"seq": self.xs[index]}]
        ys = [np.ones_like(self.xs[index])]

        return [x_ids, xs, ys]
