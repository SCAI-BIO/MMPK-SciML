# MMPK-SciML

## Description

Official repository for the *Scientific Machine learning for predicting plasma concentrations in anti-cancer therapy* paper . This repository contains the code for our MMPKsciML model and all the models tested in our [preprint]().

## Download

To download everything from this repository onto your local directory, execute the following line on your terminal:
```
$ git clone  MMPKSciML
$ cd MMPKSciML
```

## Data

Data can be made available upon reasonable request. Please contact the Department of Clinical Pharmacy at the University of Bonn!

[Prof. Dr. Ulrich Jaehde](mailto:u.jaehde@uni-bonn.de)

An der Immenburg 4, D-53121 Bonn (Germany) 

+49 228 735252


# Repo organization

    ├── LICENSE
    ├── README.md                  
    ├── data
    │   ├── 5fu.csv                <- 5FU dataset.
    │   ├── Sunitinib              <- Sunitinib dataset.
    │       └── ... .csv           <- csv files
    │
    ├── models                     <- Trained models. The code generate all the folders as follows:
    │   ├── 5FU                    <- Trained models on the 5FU dataset
    │       └──Exp name            <- Folder with all the results for a specific experiment
    │   └── Sunitinib              <- Trained models on the Sunitinib dataset
    │       └──Exp name            <- Folder with all the results for a specific experiment
    │
    │
    ├── 5FU                        <- Code for all the models using the 5FU dataset
    │   ├── MMPKSCIML              <- Code for the MMPK-SciML model
    │   ├── CML                    <- Code for the classic Machine Learning models
    │   └── PopPK                  <- Code for the PopPK model
    │
    ├── Sunitinib                  <- Code for all the models using the Sunitinib dataset
    │   ├── MMPKSCIML              <- Code for the MMPK-SciML model
    │   ├── CML                    <- Code for the classic Machine Learning models
    │   └── PopPK                  <- Code for the PopPK model

--------
## Contact
- Prof. Dr. Holger Fröhlich: holger.froehlich@scai.fraunhofer.de
- Diego Valderrama: diego.felipe.valderrama.nino@scai.fraunhofer.de
- AI and Data Science Group, Bioinformatics Department, Fraunhofer Institute for Algorithms and Scientific Computing (SCAI), Schloss Birlinghoven, 1, 53757 Sankt Augustin.
