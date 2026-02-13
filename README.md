# DLEM

This package provides functions to run differentiable loop extrusion model on HiC/Micro-C experiments formated as mcool files.

## Installation
Requirements: Python >= 3.10, recent pip/setuptools/wheel.
```bash
git clone https://github.com/chikinalab/dLEMpytorch.git
cd dLEMpytorch
conda create -n dlem python=3.10
conda activate dlem
pip install -e .
```


## Usage
Deep dLEM model predicts Hi-C/Micro-C contact map structure from sequence and epigenomic features. The following provides example commands for training and doing inference with the deep dLEM model.

### dLEM training

An example command for training a dLEM model looks like this:
```
DLEM \
    /path/to/input.mcool  \ # contact map mcool file
    /path/to/output.tsv \ # output path
    10000 \ # resolution
    --stride 10000 \
    --window-size 128 \
    --model-name netdlem2 \
    --chrom-subset chr1 chr2 chr3 \
    --perc-nan-threshold 0.3 \
    --lr 0.01 \
    --reader-name datareader_cooler \
```

- `cooler_file`: Path to the mcool file.
- `output_path`: Path to the output TSV file.
- `resolution`: Resolution (in base pairs) of the contact map to train on. The specified resolution should exist in the .mcool file.
- `stride`: Step size (in base pairs) between consecutive patches along the diagonal.
- `window-size`: Algorithm fits patches along the diagonal of the contact map. This specifies the dimensions of the square patch. The default is `2MB/resolution`.
- `model-name`: Name of the model architecture to use. See different model architectures under `dlem/models`.
- `chrom-subset`: List of chromosomes to include in training.
- `perc-nan-threshold`: Maximum allowed percentage of the missing positions in a patch. Patches exceeding this threshold are skipped.
- `lr`: Learning rate for the model fit. ADAM is used.
- `reader-name`: Data reader used to load contact maps. The default reader loads data from .mcool files and returns an array. The user can provide different readers to read from different file types. See the other implemented readers under `dlem/readers`.


### dLEM inference

Inference uses a trained model checkpoint to predict a full contact map patch for a specific genomic region, using sequence and epigenomic input tracks. Find example model checkpoint files under `./model_checkpoint`.

```
dlem-inference \
    /path/to/output \ # directory to save output contactmap
    chr1 \ # chromosome to run inference on
    2805000 \ # the start coordinate
    --tracks /path/to/bigwig \ # epigenetic track bigwigs
    --seq-features /path/to/bigwig \ # sequence feature bigwigs
    --model-checkpoint /path/to/model_checkpoint \
    --device cpu \
```

- `output_dir`: Directory where the predicted contact map (.npy) will be saved.
- `chrom`: Chromosome to run inference on (e.g. chr1).
- `start`: Start coordinate (in base pairs, inclusive) of the genomic region.
- `tracks`: One or more epigenomic signal tracks in bigWig format
(e.g. DNase-seq, ATAC-seq).
- `seq-features`: One or more sequence-derived feature tracks in bigWig format
(e.g. directional CTCF motif matches).
- `model-checkpoint`: Path to the trained PyTorch Lightning checkpoint.
- `resolution`: Contact map resolution in base pairs.
- `layer-channel-num`: Number of channels for the sequence pooler.
- `bin-size`: Bin size passed to the sequence pooler.
- `data-patch-size`: Patch size used by the model.
- `data-track-dim`: Number of epigenomic track channels.
- `seq-dim`: Number of sequence feature channels.
- `data-start`: Start diagonal index used by the model.
- `data-stop`: Stop diagonal index used by the model.
- `channel-per-route`: Number of channels per route in the head network.
- `head-layer-num`: Number of layers in the head network.
- `device`: Torch device to use (e.g. 'cpu', 'cuda', or 'cuda:0').
- `bigwig-pooling`: Optional pooling resolution passed to bigWig reader. Default: no pooling.

#### Feature tracks data downloading
The example feature tracks data for performing dLEM inference are available on [zenodo](https://doi.org/10.5281/zenodo.18633326). This includes DNase tracks for H1 and HFF cell types and directional CTCF matches as sequence features. 

An example model checkpoint file can be found under `./model_checkpoint/`.

## How to cite

If you use dLEM or any of its language bindings in your research, please cite the following publication:

Tina Subic, Ali Tŭgrul Balcı, Kristina Perevoshchikova, Diego Borges-Rivera, Jieni Hu, Geoffrey Fudenberg, Jacqueline Dresch, Maria Chikina, Mechanistic Genome Folding at Scale through the Differentiable Loop Extrusion Model
_Biorxiv_, [https://www.biorxiv.org/content/10.1101/2025.10.17.682904v2](https://www.biorxiv.org/content/10.1101/2025.10.17.682904v2)