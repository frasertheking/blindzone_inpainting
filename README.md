<div align="center">

![logo](https://github.com/frasertheking/blindpaint/blob/main/images/logo.jpg?raw=true)

BlindPaint: a 3Net+ for Inpainting Radar Blind Zones, maintained by [Fraser King](https://frasertheking.com/)

![build](https://github.com/buttons/github-buttons/workflows/build/badge.svg)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

---

## Development of a deeply supervised full-scale connected U-Net for reflectivity inpainting in spaceborne radar blind zones

This project is currently being written into a journal article for [Artificial Intelligence for the Earth Systems](https://www.ametsoc.org/index.cfm/ams/publications/journals/artificial-intelligence-for-the-earth-systems/).

Snowfall is a critical contributor to the global water-energy budget, with important connections to water resource management, flood mitigation, and ecosystem sustainability. However, traditional spaceborne remote monitoring of snowfall faces challenges due to a near-surface radar blind zone, which masks a portion of the atmosphere. Here, a deep learning model was developed to fill in missing data across these regions using surface radar and atmospheric climate variables. The model accurately predicts reflectivity, with significant improvements over conventional methods. This innovative approach enhances our understanding of reflectivity patterns and atmospheric interactions, bolstering advances in remote snowfall prediction.

This repository contains the model structure, training and testing code for the project. We also include a handful of example cases and  preprocessed data on Zenodo if you wish to test the model on your own hardware.

![inpainting output](https://github.com/frasertheking/blindpaint/blob/main/images/im1.jpg?raw=true)

## Model Structure

Through an adaptation of the architecture outlined in [Huang et al., 2020](https://ieeexplore.ieee.org/document/9053405) for image segmentation, we developed a full-scale connected U-Net with deep supervision for block inpainting missing regions in vertically pointing radar. The encoder-decoder architecture allows the model to learn deep latent features in aloft cloud and surrounding climate state variables to intelligently reconstruct near-surface reflectivity values. This project also strongly builds on the findings of [Geiss and Hardin, 2021](https://amt.copernicus.org/articles/14/7729/2021/). Our 3Net+ model architecture diagram is below:

![model diagram](https://github.com/frasertheking/blindpaint/blob/main/images/im2.jpg?raw=true)

## Performance

Current pixel-scale and scene-scale accuracy levels are substantially improved over traditional linear inpainting techniques, with the largest improvements noted in multi-layer clouds, near surface reflectivity gradients, shallow snowfall and virga cases, and the prediction of mixed-phase clouds during data sparse conditions. An example performance diagram showing the detection capabilities of the model is shown below for a handful of cases:

![performance diagram](https://github.com/frasertheking/blindpaint/blob/main/images/im3.jpg?raw=true)

## Computational Costs

The UNet models we examine here have around 7 million trainable parameters. We used a combination of Azure cluster GPUs (V100s) and a high performance desktop (RTX 4090) for most training, which would take about half a day on average. Some additional performance details are included below:

![training table](https://github.com/frasertheking/blindpaint/blob/main/images/im4.jpg?raw=true)

## Installation

To perform your own training tests, you'll want to clone this repo, and create a Conda environment to install the necessary packages using the below command. Note, you will need the GPU enabled version of Tensorflow.

```bash
conda env create -f environment.yml
source activate tflow
```

## Examples

If you want to see how data is loaded, models are defined and training is performed, check out our example Jupyter Notebook (example.ipynb) which covers most of the basics to get you up and running with a [small example dataset](https://drive.google.com/drive/folders/1VpVWexR5soQkjcIWsuc6jf-hVBlgtptN?usp=sharing). 

## Structure

The repository is broken up into multiple modules, responsible for different parts of the preprocessing, training, testing, evaluating and plotting processes used throughout the project. For instance, loss functions are kept in the bp_loss.py file, while the general configuration settings are in the bp_configs.py file. Each of these are then imported when necessary into the larger run scripts to produce less coupled and more cohesive code.

## Configs

As previously mentioned, most of the code editing you'll need to do to configure the project for your own needs can be found in bbp_configs.py. This file allows you to change the number of training channels on the fly, tracking information such as the run identifier, inpainting sizes, MC dropout values and other post-processing constants. I've included some examples below.

```python
CHANNELS = 5
USE_DOP_SPW = False
RUN_CASE='128_' + str(CHANNELS) + 'chan_era5_nsa_oli_10km_dsv_long_hybrid'
RUN_CASE_DATA='128_' + str(CHANNELS) + 'chan_era5_nsa_oli_10km_dsv_long_hybrid/'
RUN_NAME='test-run-001'
DOWNFILL_SIZES = [16]
N_MC_TESTS = 50 
STD_CUTOFF = 1
PERC_CLOUD = 0.05
CLOUD_MASK = -0.5
PRECIPITATION_DBZ = -20
MAX_DBZ = 20
```

This is also where you can point to the downloaded test code on your system.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Note that, as a living project, code is not as clean as it could (should) be, and unit tests need to be produced in future iterations to maintain stability.

## Authors & Contact

- Fraser King, University of Michigan, kingfr@umich.edu
- Claire Pettersen, University of Michigan
- Chris Fletcher, University of Waterloo
- Andrew Geiss, Pacific Northwest National Laboratory


## Funding
This project was funded by NASA New (Early Career) Investigator Program (NIP) grant at the [University of Michigan](https://umich.edu).
# blindzone_inpainting
