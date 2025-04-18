import h5py
import numpy as np
import torch
from h5max import load_sparse_matrix
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pdb import set_trace


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
                try:
                    arr[i] = np.concatenate(([5], x[k], [6], [7] * l))
                except:
                    print(k, len(x[k]), l)
                    print(x[k])
                    print(batch[0][i])
                    set_trace()
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
        grouped_ribo_ids,
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
        self.grouped_ribo_ids = grouped_ribo_ids
        self.offsets = offsets
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
        if cond is None:
            self.cond = {
                "global": {},
                "grouped": {k: {} for k in grouped_ribo_ids.keys()},
            }
        else:
            self.cond = cond
        self.leaky_frac = leaky_frac
        self.collate_fn = collate_fn
        self.parallel = parallel

    def setup(self, stage=None):
        f = h5py.File(self.h5_path, "r")[self.exp_path]
        self.seqn_list = np.array(f[self.seqn_path])
        self.transcript_lens = np.array(f["transcript_len"])
        f.file.close()
        # evaluate conditions
        global_mask, global_masks, group_masks = self.evaluate_masks()
        # Identical mask over the samples applied to all datasets
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
        dummy_mask = {
            g: np.full_like(global_mask, True) for g in self.grouped_ribo_ids.keys()
        }
        if stage == "fit" or stage is None:
            print(f"--> Training seqnames: {[t.decode() for t in self.seqns['train']]}")
            print(f"--> Validation seqnames: {[t.decode() for t in self.seqns['val']]}")
            if len(self.seqns["test"]) > 0:
                print(f"--> Test seqnames: {[t.decode() for t in self.seqns['test']]}")
            # train set
            seqn_mask = np.isin(self.seqn_list, np.array(self.seqns["train"]))
            mask = np.logical_and(global_mask, seqn_mask)
            self.tr_idx, self.tr_len, self.tr_idx_adj, self.train_groups = (
                self.prepare_sets(mask, group_masks)
            )
            print(f"--> Training set transcripts: {len(self.tr_idx)}")
            assert len(self.tr_idx) > 0, "No transcripts in training data"
            # validation set
            seqn_mask = np.isin(self.seqn_list, self.seqns["val"])
            if self.strict_validation:
                # global_masks[0] is transcript length mask
                val_mask = np.logical_and(seqn_mask, global_masks[0])
                self.val_idx, self.val_len, self.val_idx_adj, self.val_groups = (
                    self.prepare_sets(val_mask, dummy_mask)
                )
            else:
                val_mask = np.logical_and(global_mask, seqn_mask)
                self.val_idx, self.val_len, self.val_idx_adj, self.val_groups = (
                    self.prepare_sets(val_mask, group_masks)
                )
            print(f"--> Validation set transcripts: {len(self.val_idx)}")
            assert len(self.val_idx) > 0, "No transcripts in validation data"
        if stage in ["test", "predict"] or stage is None:
            print(f"--> Test seqnames: {[t.decode() for t in self.seqns['test']]}")
            seqn_mask = np.isin(self.seqn_list, self.seqns["test"])
            # Only mask transcript lengths instead of all
            test_mask = np.logical_and(seqn_mask, global_masks[0])
            self.te_idx, self.te_len, self.te_idx_adj, self.te_groups = (
                self.prepare_sets(test_mask, dummy_mask)
            )
            print(f"--> Test set transcripts: {len(self.te_idx)}")
            assert len(self.te_idx) > 0, "No transcripts in test data"

    def evaluate_masks(self):
        """
        Evaluate and generate masks for filtering data based on global and grouped conditions.

        This method applies filtering conditions defined in the `self.cond` configuration
        to the transcript sequence data (f) and riboseq data (r). It generates masks that determine
        which samples pass the filtering criteria. The filtering is performed at two level, globally
        and for grouped samples.

        Returns:
            tuple:
                - global_mask (numpy.ndarray): A boolean mask representing the combined
                  filtering conditions applied globally across all samples.
                - global_masks (list of numpy.ndarray): A list of boolean masks, each
                  corresponding to a specific global condition.
                - group_masks (list of numpy.ndarray): A list of boolean masks, each
                  representing the combined filtering conditions applied to a specific
                  group of datasets.

        Notes:
            - The `self.cond` attribute is expected to contain two keys: "global" and
              "grouped". Each key maps to a dictionary where the keys are feature names
              and the values are functions that define the filtering conditions.
            - The `self.leaky_frac` attribute allows a fraction of samples to bypass
              the filtering conditions, except for conditions based on "transcript_len".
            - If `self.parallel` is True, the method reads grouped data in parallel
              from multiple HDF5 files. Otherwise, it reads the data sequentially.
            - The method assumes that grouped data is stored under the path
              "riboseq/{ribo_id}/5/{key}".

        """
        f = h5py.File(self.h5_path, "r")[self.exp_path]
        global_masks = []
        for key, cond_f in self.cond["global"].items():
            mask = cond_f(np.array(f[key]))
            # leaky frac allows percentage-wise randomly selected samples to be included
            # in the training set, even if they do not pass the filtering condition
            # excluding transcript length mask
            if (key != "transcript_len") and (self.leaky_frac > 0):
                prob_mask = np.random.uniform(size=len(mask)) > (1 - self.leaky_frac)
                mask[prob_mask] = True
            global_masks.append(mask)
        global_mask = np.logical_and.reduce(global_masks)
        # Masks over the samples applied per group of datasets
        group_masks = {}
        print(self.cond)
        for group, cond_dict in self.cond["grouped"].items():
            group_mask = [np.full_like(global_mask, True)]
            for key, cond_f in cond_dict.items():
                # collect read counts for all samples in group
                if self.parallel:
                    read_data = []
                    for ribo_id in self.grouped_ribo_ids[group]:
                        h5_ribo_path = f"{self.h5_path.split('.h5')[0]}_{ribo_id}.h5"
                        r = h5py.File(h5_ribo_path, "r")[self.exp_path]
                        read_data.append(np.array(r[f"riboseq/{ribo_id}/5/{key}"]))
                        r.file.close()
                else:
                    read_data = [
                        np.array(f[f"riboseq/{ribo_id}/5/{key}"])
                        for ribo_id in self.grouped_ribo_ids[group]
                    ]
                grouped_feature = np.add.reduce(read_data)
                # apply function given in config file
                mask = cond_f(grouped_feature)
                # if data filtering is not based on transcript length,
                # allow leaky filtering
                if (key != "transcript_len") and (self.leaky_frac > 0):
                    prob_mask = np.random.uniform(size=len(mask)) > (
                        1 - self.leaky_frac
                    )
                    mask[prob_mask] = True
                group_mask.append(mask)
            group_masks[group] = np.logical_and.reduce(group_mask)

        return global_mask, global_masks, group_masks

    def prepare_sets(self, mask, group_conds):
        """
        Prepares and organizes subsets of data based on a given mask and group conditions.

        Args:
            mask (numpy.ndarray): A boolean array combining global mask and contig mask.
            group_conds (dict): A list of boolean arrays representing
                group-specific conditions to be applied in conjunction with the mask.

        Returns:
            tuple: A tuple containing:
                - idxs_sorted (numpy.ndarray): Indices of the filtered data, sorted by their lengths.
                - lens_sorted (numpy.ndarray): Lengths of the filtered data, sorted in ascending order.
                - total_count (int): Total number of elements in the original mask.
                - idx_group_order (list): A list of group names in the order they were evaluated.

        Notes:
            - For each sample group, a unique mask is created which is concatened in order of evaluation
            - The array of idxs points to all transcripts that have passed the filter (i.e. True's in the concatenated mask.)
            - When initializing the data loader object, these idxs are bucketed according to transcript lens
            - The sample and exact transcript are derived from the idx value using the order in which
            the sample groups were evaluated and unit length of masks (equal across all sample groups)
        """
        mask_set = []
        len_set = []
        idx_group_order = []
        if len(group_conds) > 0:
            for group, cond_mask in group_conds.items():
                mask_set.append(np.logical_and(cond_mask, mask))
                len_set.append(self.transcript_lens[mask_set[-1]])
                # For certainty, I'm not assuming a static order in evaluation of dicts
                idx_group_order.append(group)
        else:
            mask_set.append(mask)
            len_set.append(self.transcript_lens[mask_set[-1]])
            idx_group_order.append("seq")

        mask_all = np.hstack(mask_set)
        lens = np.hstack(len_set)
        idxs = np.where(mask_all)[0]
        sort_idxs = np.argsort(lens)

        return idxs[sort_idxs], lens[sort_idxs], len(mask), idx_group_order

    def get_dataloader(self, stage):
        """
        Creates and returns a DataLoader for the specified stage (train, val, test, or predict).

        Args:
            stage (str): The stage for which the DataLoader is to be created.
                         Must be one of 'train', 'val', 'test', or 'predict'.

        Returns:
            DataLoader: A PyTorch DataLoader instance configured for the specified stage.

        Notes:
            - The `bucket` function is used to group data into batches based on
              memory and transcript constraints.
            - For the 'train' stage, the `local_shuffle` function is applied to
              shuffle the training indices and lengths locally before bucketing.
            - The `h5pyDatasetBatches` class is used to handle the dataset
              batches, which reads data from HDF5 files and other associated
              paths.
            - The DataLoader is configured with a batch size of 1.
        """
        if stage == "train":
            indices, lengths = local_shuffle(self.tr_idx, self.tr_len)
            idx_group_order = self.train_groups
            idx_adj = self.tr_idx_adj
        elif stage == "val":
            indices, lengths, idx_group_order = (
                self.val_idx,
                self.val_len,
                self.val_groups,
            )
            idx_adj = self.val_idx_adj
        elif stage in ["test", "predict"]:
            indices, lengths, idx_group_order = self.te_idx, self.te_len, self.te_groups
            idx_adj = self.te_idx_adj
        else:
            raise ValueError(
                f"Invalid stage: {stage}. Must be 'train', 'val', 'test', or 'predict'."
            )

        batches = bucket(
            indices,
            lengths,
            self.max_memory,
            self.max_transcripts_per_batch,
            stage if stage != "predict" else "test",
        )
        return DataLoader(
            h5pyDatasetBatches(
                self.h5_path,
                self.y_path,
                self.tr_id_path,
                self.use_seq,
                self.grouped_ribo_ids,
                self.offsets,
                idx_adj,
                batches,
                idx_group_order,
                self.parallel,
            ),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=(stage == "train"),
            batch_size=1,
        )

    # train_dataloader, val_dataloader, test_dataloader, predict_dataloader for pytorch lightning:
    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def predict_dataloader(self):
        return self.get_dataloader("predict")


class h5pyDatasetBatches(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading batches of data from HDF5 files. This class supports
    both sequential and ribosome profiling data, with options for parallel processing and
    dataset-specific adjustments.

    Attributes:
        h5_path (str): Path to the main HDF5 file containing the data.
        y_path (str): Path within the HDF5 file to the target labels.
        tr_id_path (str): Path within the HDF5 file to the transcript IDs.
        use_seq (bool): Flag indicating whether to use sequence data or ribosome profiling data.
        grouped_ribo_ids (dict): Dictionary mapping group names to lists of ribosome profiling sample IDs.
        offsets (dict): Dictionary specifying positional offsets for ribosome profiling data.
        idx_adj (int): Adjustment factor for indexing when multiple datasets are concatenated.
        batches (list): List of batches, where each batch contains indices of data samples.
        parallel (bool): Flag indicating whether to use parallel processing for loading data.

    Methods:
        __len__():
            Returns the number of batches in the dataset.

        open_hdf5():
            Opens the HDF5 files for reading. If parallel processing is enabled, it opens
            separate files for each ribosome profiling experiment.

        close_hdf5():
            Closes the HDF5 files that were opened for reading.

        __getitem__(index):
            Retrieves a batch of data corresponding to the given index. The data includes
            transcript IDs, input features (sequence or ribosome profiling data), and target labels.
    """

    def __init__(
        self,
        h5_path,
        y_path,
        tr_id_path,
        use_seq,
        grouped_ribo_ids,
        offsets,
        idx_adj,
        batches,
        idx_group_order,
        parallel,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.y_path = y_path
        self.tr_id_path = tr_id_path
        self.use_seq = use_seq
        self.grouped_ribo_ids = grouped_ribo_ids
        self.offsets = offsets
        self.idx_adj = idx_adj
        self.batches = batches
        self.idx_group_order = idx_group_order
        self.parallel = parallel

    def __len__(self):
        return len(self.batches)

    def open_hdf5(self):
        if self.parallel:
            h5_base = self.h5_path.split(".h5")[0]
            # flatten nested list
            sample_ids = sum(self.grouped_ribo_ids.values(), [])
            self.r = {
                sample: h5py.File(f"{h5_base}_{sample}.h5", "r")["transcript"]
                for sample in sample_ids
            }
        self.f = h5py.File(self.h5_path, "r")["transcript"]

    def close_hdf5(self):
        if self.parallel:
            for id in self.r.keys():
                self.r[id].file.close()
        self.f.file.close()

    def __getitem__(self, index):
        if not hasattr(self, "f"):
            self.open_hdf5()
        # aggregation is performed when a sample is requested
        x_ids = []
        xs = []
        ys = []
        # For every dataloader index in batch
        for dl_idx in self.batches[index]:
            # batches are stacked across all samples
            # get adjusted set_idx if multiple datasets are used
            set_idx = int(dl_idx % self.idx_adj)
            group = self.idx_group_order[dl_idx // self.idx_adj]
            x_dict = {}
            # get seq data
            if self.use_seq:
                x_dict["seq"] = self.get_seq_data(set_idx)
            # get ribo data
            else:
                x_dict["ribo"] = self.get_ribo_data(set_idx, group)

            # get transcript IDs
            x_ids.append(
                group.encode() + "|".encode() + self.f[self.tr_id_path][set_idx]
            )
            xs.append(x_dict)
            ys.append(self.f[self.y_path][set_idx])

        return [x_ids, xs, ys]

    def get_seq_data(self, idx):
        return self.f["seq"][idx]

    def get_ribo_data(self, idx, group):
        # determine data set from grouped (concatenated) datasets
        x_merge = []
        ribo_ids = self.grouped_ribo_ids[group]
        # iterate ribo-seq experiments in merged (summed) datasets
        for i, sample in enumerate(ribo_ids):
            ribo_path = f"riboseq/{sample}/5"
            if self.parallel:
                x = load_sparse_matrix(self.r[sample][ribo_path], idx, format="csr").T
            else:
                x = load_sparse_matrix(self.f[ribo_path], idx, format="csr").T
            # Legacy code for paper, instead of using transcript_pos x read length
            # matrix as input, sum reads across read lengths after shifting by
            # the predetermined offset values
            if self.offsets is not None:
                for col_i, (_, shift) in enumerate(self.offsets[sample].items()):
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
            return x / np.maximum(x.max(), 1)
        else:
            # normalize including all positions,reads
            return x / np.maximum(np.sum(x, axis=1).max(), 1)


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
