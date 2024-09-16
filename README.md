# DLEM

This package provides functions to run differentiable loop extrusion model on HiC/Micro-C experiments formated as mcool files.

## Terminal interface 

Usage:
```
DLEM [-h] [--stride STRIDE] [--window-size WINDOW_SIZE] [--model-name MODEL_NAME] [--chrom-subset CHROM_SUBSET [CHROM_SUBSET ...]]
            [--perc-nan-threshold PERC_NAN_THRESHOLD] [--lr LR] [--reader-name READER_NAME]
            cooler_file output_path resolution
```

- cooler_file: Specifies path to the mcool file.
- output_path: Specifies path to the TSV file.
- resolution: Specifies the resolution of the contact map. The corresponding map should exist in mcool.
- WINDOW_SIZE: Algorithm fits patches along the diagonal of the contact map. This specifies the dimensions of the square patch. The default is `2MB/resolution`.
- MODEL_NAME: Specifies a model other than the default DLEM model to fit to the data. See models under `dlem/models`.
- CHROM_SUBSET: If given the model only fits the contact map from these chromosomes.
- PERC_NAN_THRESHOLD: If the percentage of the missing positions in the patch exceeds this threshold the patch is skipped.
- LR: learning rate for the model fit. ADAM is used.
- READER_NAME: Default reader reads from mcool files and returns an array. The user can provide different readers to read from different file types. See the readers under `dlem/readers`. 