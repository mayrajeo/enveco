# Enveco
> ENVECO is a one-year project financed by Eurostat grants led by Finnish Environment Institute (SYKE) in partnership with Natural Resources Institute Finland (Luke). The first aim of the project is to develop novel remote sensing and machine learning methods for accounting of forest-related ecosystem services in the SEEA-EEA (System of Environmental-Economic Accounting – Experimental Ecosystem Accounting) framework.


This repository contains the code for WP3: Machine Learning and Remote Sensing in Supporting Ecosystem Accounting. The objectives for this WP are:

1. To test provisioning of remote sensing based data for the indicators of forest-related ecosystem services based on the framework defined by Mononen et al (2016) and further linked with SEEA-EEA framework by LAI et al (2018).
2. For the prediction of wall-to-wall maps, we will test if introducing deep learning methods and new temporal, spectral and spatial features will improve the prediction results compared to traditional RS modeling methods used in NFI
3. To address the uncertainties of the produed wall-to-wall maps, we will assess specifically the accuracy of locating spots that are important for ecosystem services supply.
4. To evaluate value of wooded land by utilizing ecosystem indicator data with uncertainty provided by previous tasks. Valuation methods suggested in SEEA-EEA are compared with the ones commonly applied for forest land.

## Install

Instructions here.

### Using singularity containers in Puhti

Easiest way to use and develop this library in Puhti is to use singularity container with all required libraries. 

Use instructions in [https://cloud.sylabs.io/builder] to set access tokens and then run 

```bash
singularity build --remote enveco-container.sif enveco-container.def
```

Edit `enveco-container.def` to such that correct shared locations are used, and edit row `mkdir -p /projappl /scratch /users/mayrajan` to contain your own home directory.

To start working, run singularity shell with command 

```bash
singularity shell -nv --writable-tmpfs --bind $PROJAPPL:/projappl --bind $SCRATCH:/scratch --bind $HOME:/users/<uid> enveco-container.sif
```

## Usage

Add examples.

## Authors

Add here.
