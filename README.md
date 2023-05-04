<div align="center">
<h1>transcript_transformer</h1> 

Deep learning utility functions for processing and annotating transcript genome data.

[![PyPi Version](https://img.shields.io/pypi/v/transcript-transformer.svg)](https://pypi.python.org/pypi/transcript-transformer/)
[![GitHub license](https://img.shields.io/github/license/jdcla/transcript_transformer)](https://github.com/jdcla/transcript_transformer/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/jdcla/transcript_transformer)](https://github.com/jdcla/transcript_transformer/issues)
[![GitHub stars](https://img.shields.io/github/stars/jdcla/transcript_transformer)](https://github.com/jdcla/transcript_transformer/stargazers)
</div>


`transcript_transformer`  is constructed in concordance with the creation of TIS Transformer, ([paper](https://doi.org/10.1093/nargab/lqad021), [repository](https://github.com/jdcla/TIS_transformer)) and RIBO-former (to be released). `transcript_transformer` makes use of the [Performer](https://arxiv.org/abs/2009.14794) architecture to allow for the annotations and processing of transcripts at single nucleotide resolution. The package makes use of `h5py` for data loading and `pytorch-lightning` as a high-level interface for training and evaluation for deep learning models. `transcript_transformer` is designed to allow a high degree of modularity, but has not been tested for every combination of arguments, and can therefore return errors.

## 🔗 Installation
`pytorch` needs to be separately [installed by the user](https://pytorch.org/get-started/locally/). 

Next, the package can be installed running 
```bash
pip install transcript-transformer
```

## 📖 User guide <a name="code"></a>

The library features a tool that can be called directly by the command `transcript_transformer`, featuring three main functions: `pretrain`, `train` and `predict`. Data loading is achieved using the `h5` format, handled by the `h5py` package. A introductary explanation on how to apply models for predicting TIS using sequence information or ribosome profiling data, please refer to the respective repositories of [TIS Transformer](https://github.com/jdcla/TIS_transformer) and RIBO-former (Soon).

### Data loading
Information is separated by transcript and information type. Information belonging to a single transcript is mapped according to the index they populate within each `h5py.dataset`, used to store different types of information. [Variable length arrays](https://docs.h5py.org/en/stable/special.html#arbitrary-vlen-data) are used to store the sequences and annotations of all transcripts under a single data set. 
Sequences are stored using integer arrays following: `{A:0, T:1, C:2, G:3, N:4}`
An example `data.h5` has the following structure:


```
data.h5                                     (h5py.file)
    transcript                              (h5py.group)
    ├── tis                                 (h5py.dataset, dtype=vlen(int))
    ├── contig                              (h5py.dataset, dtype=str)
    ├── id                                  (h5py.dataset, dtype=str)
    ├── seq                                 (h5py.dataset, dtype=vlen(int))
    ├── ribo                                (h5py.group)
    │   ├── SRR0000001                      (h5py.group)
    │   │   ├── 5                           (h5py.group)
    │   │   │   ├── data                    (h5py.dataset, dtype=vlen(int))
    │   │   │   ├── indices                 (h5py.dataset, dtype=vlen(int))
    │   │   │   ├── indptr                  (h5py.dataset, dtype=vlen(int))
    │   │   │   ├── shape                   (h5py.dataset, dtype=vlen(int))
    │   ├── ...
    │   ....
    
```

Ribosome profiling data is saved by reads mapped to each transcript position. Mapped reads are furthermore separated by their read lengths. As ribosome profiling data is often sparse, we made use of `scipy.sparse` to save data within the `h5` format. This allows us to save space and store matrix objects as separate arrays. Saving and loading of the data is achieved using the [``h5max``](https://github.com/jdcla/h5max) package.

<div align="center">
<img src="https://github.com/jdcla/h5max/raw/main/h5max.png" width="600">
</div>

Dictionary `.json` files are used to specify the application of data to `transcript_transformer`. When no sequence information or ribosome profiling data is used, either entry `seq` or `ribo` is set to `false`. For each ribosome profiling dataset, custom [P-site offsets](https://plastid.readthedocs.io/en/latest/glossary.html#term-P-site-offset) can be set per read length. 

```json
{
  "h5_path":"data.h5",
  "exp_path":"transcript",
  "y_path":"tis",
  "chrom_path":"contig",
  "id_path":"id",
  "seq":"seq",
  "ribo": {
    "SRR000001/5": {
      "25": 7,
      "26": 7,
      "27": 8,
      "28": 10,
      "29": 10,
      "30": 11,
      "31": 11,
      "32": 7,
      "33": 7,
      "34": 9,
      "35": 9,
    }
  }
}
```



### pretrain

Conform with transformers trained for natural language processing objectives, models can first be trained following a self-supervised learning objective. Using a masked language modelling approach, models are tasked to predict the classes of the masked input tokens. As such, a model is trained the 'semantics' of transcript sequences. The approach is similar to the one described by [Zaheer et al. ](https://arxiv.org/abs/2007.14062). This approach has not been using ribosome profiling data.


```
transcript_transformer pretrain -h

positional arguments:
  dict_path             dictionary (json/yaml) path containing input data file info

options:
  -h, --help            show this help message and exit
  --train str [str ...]
                        contigs in data_path folder used for training. If not specified, training is performed on all available contigs excluding val/test contigs (default: None)
  --val str [str ...]   contigs in data_path folder used for validation (default: None)
  --test str [str ...]  contigs in data_path folder used for testing (default: None)
  --ribo_offset boolean
                        offset mapped ribosome reads by read length (default: False)
  --name str            name of the model (default: )
  --log_dir str         log dir (default: lightning_logs)
```

Example

```
transcript_transformer pretrain input_data.json --val 1 13 --test 2 14 --max_epochs 70 --gpu 1 
```

</details>

### train
The package supports training the models architectures listed under `transcript_transformer/models.py`. The function expects a `.json` file containing the input data info (see [data loading](https://github.com/jdcla/transcript_transformer#data-loading)). It is possible to start training upon pre-trained models using the `--transfer_checkpoint` functionality.


```
transcript_transformer train -h
    
positional arguments:
  dict_path             dictionary (json/yaml) path containing input data file info

options:
  -h, --help            show this help message and exit
  --train str [str ...]
                        contigs in data_path folder used for training. If not specified, training is performed on all available contigs excluding val/test contigs (default: None)
  --val str [str ...]   contigs in data_path folder used for validation (default: None)
  --test str [str ...]  contigs in data_path folder used for testing (default: None)
  --ribo_offset boolean
                        offset mapped ribosome reads by read length (default: False)
  --name str            name of the model (default: )
  --log_dir str         log dir (default: lightning_logs)
```

Example

```
transcript_transformer train input_data.json --val 1 13 --test 2 14 --max_epochs 70 --transfer_checkpoint lightning_logs/mlm_model/version_0/ --name experiment_1 --gpu 1 
```

### predict

The predict function returns probabilities for all nucleotide positions on the transcript and can be saved using the `.npy` or `.h5` format. In addition to reading from `.h5` files, the function supports the use of a single RNA sequence as input or a path to a `.fa` file. Note that `.fa` and `.npy` formats are only supported for models that only apply transcript nucleotide information.

```
transcript_transformer predict -h

positional arguments:
  input_data            path to json/yaml dict (h5) or fasta file, or RNA sequence
  input_type            type of input
  checkpoint            path to checkpoint of trained model

options:
  -h, --help            show this help message and exit
  --prob_th PROB_TH     minimum prediction threshold at which additional information is processed (default: 0.01)
  --save_path save_path
                        save file path (default: results)
  --output_type {npy,h5}
                        file type of raw model predictions (default: npy)
```


Example

```
transcript_transformer predict AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACGGT RNA --output_type npy models/example_model.ckpt
transcript_transformer predict data/example_data.fa fa --output_type npy models/example_model.ckpt
```

### Output data

The model returns predictions for every nucleotide on the transcripts. For each transcript, the array lists the transcript label and model outputs. The tool can output predictions using both the `npy` or `h5` format. 

```python
>>> results = np.load('results.npy', allow_pickle=True)
>>> results[0]
array(['>ENST00000410304',
       array([2.3891837e-09, 7.0824785e-07, 8.3791534e-09, 4.3269135e-09,
              4.9220684e-08, 1.5315813e-10, 7.0196869e-08, 2.4103475e-10,
              4.5873511e-10, 1.4299616e-10, 6.1071654e-09, 1.9664975e-08,
              2.9255699e-07, 4.7719610e-08, 7.7600065e-10, 9.2305236e-10,
              3.3297397e-07, 3.5771163e-07, 4.1942007e-05, 4.5123262e-08,
              1.0270607e-11, 1.1841109e-09, 7.9038587e-10, 6.5511790e-10,
              6.0892291e-13, 1.6157842e-11, 6.9130129e-10, 4.5778301e-11,
              2.1682500e-03, 2.3315516e-09, 2.2578116e-11], dtype=float32)],
      dtype=object)

```

### Other function flags
Various other function flags dictate the properties of the dataloader, model architecture and training procedure.

<details><summary>Pytorch lightning trainer</summary>

```
pl.Trainer:
  --accelerator {cpu,gpu,tpu,ipu,hpu,mps,auto}
                        computational hardware to apply (default: cpu)
  --strategy str        strategy for multi-gpu computation (default: auto)
  --devices int [int ...]
                        device to use (default: 0)
  --max_epochs int      maximum epochs of training (default: 60)

```

</details>

<details><summary>Dataloader</summary>

```
data loader arguments

  --min_seq_len int     minimum sequence length of transcripts (default: 0)
  --max_seq_len int     maximum sequence length of transcripts (default: 30000)
  --leaky_frac float    fraction of samples that escape conditions (ribo-seq) (default: 0.05)
  --num_workers int     number of data loader workers (default: 12)
  --max_memory int      MB value applied for bucket batches based on rough estimates (default: 24000)
  --max_transcripts_per_batch int
                        maximum of transcripts per batch (default: 2000)
```

</details>

<details><summary>Model architecture</summary>

```
Model:
  Transformer arguments

  --transfer_checkpoint str
                        Path to checkpoint pretrained model (default: None)
  --lr float            learning rate (default: 0.001)
  --decay_rate float    multiplicatively decays learning rate for every epoch (default: 0.96)
  --warmup_steps int    number of warmup steps at the start of training (default: 1500)
  --num_tokens int      number of unique nucleotide input tokens (default: 5)
  --dim int             dimension of the hidden states (default: 30)
  --depth int           number of layers (default: 6)
  --heads int           number of attention heads in every layer (default: 6)
  --dim_head int        dimension of the attention head matrices (default: 16)
  --nb_features int     number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head (default: 80)
  --feature_redraw_interval int
                        how frequently to redraw the projection matrix (default: 1000)
  --generalized_attention boolean
                        applies generalized attention functions (default: True)
  --kernel_fn boolean   generalized attention function to apply (if generalized attention) (default: ReLU())
  --reversible boolean  reversible layers, from Reformer paper (default: False)
  --ff_chunks int       chunk feedforward layer, from Reformer paper (default: 1)
  --use_scalenorm boolean
                        use scale norm, from 'Transformers without Tears' paper (default: False)
  --use_rezero boolean  use rezero, from 'Rezero is all you need' paper (default: False)
  --ff_glu boolean      use GLU variant for feedforward (default: False)
  --emb_dropout float   embedding dropout (default: 0.1)
  --ff_dropout float    feedforward dropout (default: 0.1)
  --attn_dropout float  post-attn dropout (default: 0.1)
  --local_attn_heads int
                        the amount of heads used for local attention (default: 4)
  --local_window_size int
                        window size of local attention (default: 256)
  --debug boolean       debug mode disables logging and checkpointing (only for train) (default: False)
  --patience int        Number of epochs required without the validation loss reducingto stop training (default: 8)
  --mask_frac float     fraction of inputs that are masked, only for self-supervised training (default: 0.85)
  --rand_frac float     fraction of inputs that are randomized, only for self-supervised training (default: 0.1)
  --metrics [{ROC,PR} ...]
                        metrics calculated at the end of the epoch for the validation/testset. These bring a cost to memory (default: ['ROC', 'PR'])
```

</details>

## ✔️ Package features

- [ ] creation of `h5` file from genome assemblies and ribosome profiling datasets
- [x] bucket sampling
- [x] pre-training functionality
- [x] data loading for sequence and ribosome data
- [x] custom target labels
- [ ] function hooks for custom data loading and pre-processing
- [x] model architectures
- [x] application of trained networks
- [ ] test scripts

## 🖊️ Citation <a name="citation"></a>
       
```bibtex
@article {10.1093/nargab/lqad021,
    author = {Clauwaert, Jim and McVey, Zahra and Gupta, Ramneek and Menschaert, Gerben},
    title = "{TIS Transformer: remapping the human proteome using deep learning}",
    journal = {NAR Genomics and Bioinformatics},
    volume = {5},
    number = {1},
    year = {2023},
    month = {03},
    issn = {2631-9268},
    doi = {10.1093/nargab/lqad021},
    url = {https://doi.org/10.1093/nargab/lqad021},
    note = {lqad021},
    eprint = {https://academic.oup.com/nargab/article-pdf/5/1/lqad021/49418780/lqad021\_supplemental\_file.pdf},
}
```
