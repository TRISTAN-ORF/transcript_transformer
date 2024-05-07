<div align="center">
<h1>transcript_transformer</h1> 

Deep learning utility functions for processing and annotating transcript genome data.

[![PyPi Version](https://img.shields.io/pypi/v/transcript-transformer.svg)](https://pypi.python.org/pypi/transcript-transformer/)
[![GitHub license](https://img.shields.io/github/license/TRISTAN-ORF/transcript_transformer)](https://github.com/TRISTAN-ORF/transcript_transformer/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/TRISTAN-ORF/transcript_transformer)](https://github.com/TRISTAN-ORF/transcript_transformer/issues)
[![GitHub stars](https://img.shields.io/github/stars/TRISTAN-ORF/transcript_transformer)](https://github.com/TRISTAN-ORF/transcript_transformer/stargazers)
</div>


`transcript_transformer`  is constructed in concordance with the creation of TIS Transformer, ([paper](https://doi.org/10.1093/nargab/lqad021), [repository](https://github.com/TRISTAN-ORF/TIS_transformer)) and RIBO-former ([paper](https://doi.org/10.1101/2023.06.20.545724), [repository paper](https://github.com/TRISTAN-ORF/RiboTIE_article), [repository tool](https://github.com/TRISTAN-ORF/RiboTIE)). `transcript_transformer` makes use of the [Performer](https://arxiv.org/abs/2009.14794) architecture to allow for the annotations and processing of transcripts at single nucleotide resolution. The package applies `h5py` for data loading and `pytorch-lightning` as a high-level interface for training and evaluation of deep learning models. `transcript_transformer` is designed to allow a high degree of modularity, but has not been tested for every combination of arguments, and can therefore return errors. For a more targeted and streamlined explanation on how to apply TIS transformer or RIBO-former, please refer to their repositories.

## 🔗 Installation
`pytorch` needs to be separately [installed by the user](https://pytorch.org/get-started/locally/). 

Next, the package can be installed running 
```bash
pip install transcript-transformer
```

## 📖 User guide <a name="code"></a>

The library features a tool that can be called directly by the command `transcript_transformer`, featuring four main functions: `data`, `pretrain`, `train` and `predict`.  

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


### data

`transcript_transformer data` is used to process the transcriptome of a given assembly to make it readily available for data loading. [Dictionary `.yml`/`.json`](https://github.com/TRISTAN-ORF/transcript_transformer/blob/main/template.yml) files are used to specify the application of data to the models. After processing, given dictionary files can still be altered to define what data is used for a specific run. As such, for a given assembly, it is possible to store all available data in a single database. New ribosome profiling experiments can be added to an existing database by running `transcript_transformer data` again after updating the config file.

The following command can be used to parse data by running:
```bash
transcript_transformer data template.yml
```
where `template.yml` is:
```yaml
gtf_path : path/to/gtf_file.gtf
fa_path : path/to/fa_file.fa
########################################################
## add entries when using ribosome profiling data.
## format: 'id : ribosome profiling paths'
## leave empty for sequence input models (TIS transformer)
## DO NOT change id after data is parsed to h5 file
########################################################
ribo_paths :
  SRR000001 : ribo/SRR000001.sam
  SRR000002 : ribo/SRR000002.sam
  SRR000003 : ribo/SRR000003.sam
########################################################
## Data is parsed and stored in a hdf5 format file.
########################################################
h5_path : my_experiment.h5
```

Several other options exist that specify how ribosome profiling data is loaded. Refer to [`template.yml`](https://github.com/TRISTAN-ORF/transcript_transformer/blob/main/template.yml), available in the root directory of this repository, for more information on each option. 

### pretrain

Conform with transformers trained for natural language processing objectives, models can first be trained following a self-supervised learning objective. Using a masked language modelling approach, models are tasked to predict the classes of the masked input tokens. As such, a model is trained the 'semantics' of transcript sequences. The approach is similar to the one described by [Zaheer et al. ](https://arxiv.org/abs/2007.14062).


Example

```
transcript_transformer pretrain input_data.yml --val 1 13 --test 2 14 --max_epochs 70 --accelerator gpu --devices 1
```

</details>

### train
The package supports training the models architectures listed under `transcript_transformer/models.py`. The function expects the configuration file containing the input data info (see [data loading](https://github.com/TRISTAN-ORF/transcript_transformer#data)). Use the `--transfer_checkpoint` flag to start training upon pre-trained models.



Example

```
transcript_transformer train input_data.yml --val 1 13 --test 2 14 --max_epochs 70 --transfer_checkpoint lightning_logs/mlm_model/version_0/ --name experiment_1 --accelerator gpu --devices 1
```

### predict

The predict function returns probabilities for all nucleotide positions on the transcript and can be saved using the `.npy` or `.h5` format. In addition to reading from `.h5` files, the function supports the use of a single RNA sequence as input or a path to a `.fa` file. Note that `.fa` and `.npy` formats are only supported for models that only apply transcript nucleotide information.


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
Various other function flags dictate the properties of the dataloader, model architecture and training procedure. Check them out 

```bash
transcript_transformer data -h 
transcript_transformer pretrain -h 
transcript_transformer data -h
transcript_transformer predict -h 
```


</details>

## ✔️ Package features

- [x] creation of `h5` file from genome assemblies and ribosome profiling datasets
- [x] bucket sampling
- [x] pre-training functionality
- [x] data loading for sequence and ribosome data
- [x] custom target labels
- [ ] function hooks for custom data loading and pre-processing
- [x] model architectures
- [x] application of trained networks
- [ ] post-processing
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
