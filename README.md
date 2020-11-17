# Enveco
> ENVECO is a one-year project financed by Eurostat grants led by Finnish Environment Institute (SYKE) in partnership with Natural Resources Institute Finland (Luke). The first aim of the project is to develop novel remote sensing and machine learning methods for accounting of forest-related ecosystem services in the SEEA-EEA (System of Environmental-Economic Accounting – Experimental Ecosystem Accounting) framework.


This repository contains the code for WP3: Machine Learning and Remote Sensing in Supporting Ecosystem Accounting. 

## Install

TODO add `requirements.txt` and `environment.yml`.

First, clone the repository

```bash
git clone https://github.com/jaeeolma/enveco
```

Then install the requirements with 

```bash
pip install -r requirements.txt
```

or with 

```bash
conda env create -f environment.yml
```

Recommended way is with conda, because installing Gdal can be troublesome.

### With singularity containers in Puhti

Easiest way to use and develop this library in Puhti is to use singularity container with all required libraries. 

Use instructions in [https://cloud.sylabs.io/builder](https://cloud.sylabs.io/builder) to set access tokens and then run 

```bash
singularity build --remote enveco-container.sif enveco-container.def
```

Edit `enveco-container.def` to such that correct shared locations are used, and edit row `mkdir -p /projappl /scratch /users/mayrajan` to contain your own home directory.

To start working, run singularity shell with command 

```bash
singularity shell -nv --writable-tmpfs --bind $PROJAPPL:/projappl --bind $SCRATCH:/scratch --bind $HOME:/users/<uid> enveco-container.sif
```

## Usage

Example workflows are show in [Examples](https://github.com/jaeeolma/enveco/tree/master/examples). 

[Predict volume from LiDAR features](https://github.com/jaeeolma/enveco/tree/master/examples/Predict%20volume%20from%20LiDAR%20features.ipynb) shows how to use ANN and Random Forest for predictions, and [Voxelizations with 3DCnns](https://github.com/jaeeolma/enveco/blob/master/examples/Voxelizations%20with%203DCnns.ipynb) shows an example of predicting features from voxel grids.

## Authors

Todo
