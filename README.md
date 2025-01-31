# Site specific emulators for tsunami run-up simulations.

[![SQAaaS badge](https://github.com/EOSC-synergy/SQAaaS/raw/master/badges/badges_150x116/badge_software_silver.png)](https://api.eu.badgr.io/public/assertions/0TVav2TXQPCz_R7ijbeaVg "SQAaaS silver badge achieved")

[![SQAaaS badge shields.io](https://img.shields.io/badge/sqaaas%20software-silver-lightgrey)](https://api.eu.badgr.io/public/assertions/0TVav2TXQPCz_R7ijbeaVg "SQAaaS silver badge achieved")

Local Probabilistic Tsunami Hazard Analysis (PTHA) aims to quantify the probability distribution of inundation intensity parameters, such as maximum flow-depth, on a given location over a given period of time. A common workflow consists of three main stages [Gibbons 2020]: 

1. Establish a stochastic model of the tsunami sources. Based on this model a set of (weighted) scenarios is generated. 
2. For each source scenario, approximate maps of the inundation intensity parameters are calculated. 
To this end one apply depth integrated equations, i.e., the shallow water equations. Linearized shallow water equations are suitable when wave heights are small compared with the depth, however, as waves approach land, the full nonlinear shallow water equations are needed to model the run-up. 
3. Hazard aggregation.

To obtain a sufficiently accurate representation of the distribution of the source, a large number of scenarios is necessary. In such cases, a site specific tsunami runup emulator, trained on precalculated data, enables fast simulations, and hence the assessment of a large number of scenarios.

In this project, we seek to construct a sufficiently accurate emulator to meet the needs associated with PTHA and PTF (Probabilistic Tsunami Forecasting).


# Content

This repository contains the code used to build, train and test ML-models for predicting tsunami inundation maps from offshore time-series. The results are documented in the article *Machine Learning Emulation of High Resolution Inundation Maps* (Submitted to GJI).
The code is written in [Julia](https://julialang.org/) using [Flux](https://fluxml.ai/Flux.jl/stable/).

# Usage
First create a data directory for defining test and training sets. The training data is defined in a text file. Each line should contain the path (relative to a root data folder) to each scenario file.
Test data is specified similarly. The current setup assumes the data is stored in NetCDF. The loading of the data applies [NCDatasets.jl](https://juliapackages.com/p/ncdatasets) and is implemented in the `datareader.jl` module.

The analysis is divided into three main steps: 
1. **[Create]** The creation of a model is done in the notebook `create-model.ipynb`. As a final step, the model is stored as a subfolder of a `runs` directory. The model folder contains the following:
    - A Flux model stored as a `.bson` file.
    - A `config.yml` file used to store dataset configuration and training parameters.
    - The scripts used to run the training (a copy current version of `train-model.jl` and `datareader.jl`).
    
    Your folder structure should now look like:
    ```
    ├── data
    |   ├── test.txt
    |   └── train.txt
    ├── runs
    |   └── model_name
    |       ├── config.yml
    |       ├── ct_mask.txt
    |       ├── datareader.jl
    |       ├── model_name.jls
    |       ├── model_config.jl
    |       ├── optimizer.bson
    |       └── train-model.jl
    ├── notebooks
    ├── scripts
    ├── Project.toml
    ├── Manifest.jl
    └── .gitignore
    ```

2. **[Train]** To train the model apply the script ``train-model.jl``. I.e.,
    ```terminal
    [rundir/model_name]$ julia --project train-model.jl 
    ```
    Note that you can set the training parameters directly in the model folder without applying the `create-model.ipynb`. After training the summary file "train-summary.txt" is available in the model folder. 

3. **[Evaluate]** The evaluation of the model is described in `evaluate-model.ipynb`. There are also other notebooks used for evaluation and comparison of model results. Note also the `predict.jl` script used to make predictions of specific events.

### Prerequisites
- Make sure you have [julia installed](https://julialang.org/downloads/platform/). Version information is stored in `Manifest.toml`. 
- GPU. The current script runs on CUDA. However, only minor changes in `train-model.jl` is needed to run on CPU.

### Improvements.
Currently the loading of the data has been written with training sets that are too large to fit in memory in mind. Several of the applied datsets do fit in memory. It would make training much faster if we could include a switch to load the whole set into memory before training.  

### Publication.  
If you use the code, we would like you to refer to the following publication:

Briseid Storrøsten, Erlend; Ragu Ramalingam, Naveen; Lorito, Stefano;  
Volpe, Manuela; Sánchez-Linares, Carlos; Løvholt, Finn; Gibbons, Steven J (2024).  
Machine learning emulation of high resolution inundation maps,  
*Geophysical Journal International*, 
238, Issue 1, pp. 382–399,  
https://doi.org/10.1093/gji/ggae151

![Screenshot from the above paper (https://doi.org/10.1093/gji/ggae151) ](InundationPromoImage.png)

## References

- Briseid Storrøsten et al. 2024 - [Machine learning emulation of high resolution inundation maps](https://doi.org/10.1093/gji/ggae151)  
- Gibbons et al. 2020 - [Probabilistic Tsunami Hazard Analysis: High Performance Computing for Massive Scale Inundation Simulations](https://doi.org/10.3389/feart.2020.591549)
