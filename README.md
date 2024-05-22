# MMPK-SciML

## Description

Official repository for the *Scientific Machine learning for predicting plasma concentrations in anti-cancer therapy* paper . This repository contains the code for our MMPKsciML model and all the models tested in our [preprint](https://www.medrxiv.org/content/10.1101/2024.05.06.24306555v1.full.pdf+html).

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

## Citation
If this code is helpful in your research, please cite the following papers:

@article{valderrama2024integrating,
  title={Integrating machine learning with pharmacokinetic models: Benefits of scientific machine learning in adding neural networks components to existing PK models},
  
  author={Valderrama, Diego and Ponce-Bobadilla, Ana Victoria and Mensing, Sven and Fr{\"o}hlich, Holger and Stodtmann, Sven},
  
  journal={CPT: Pharmacometrics \& Systems Pharmacology},
  
  volume={13},
  
  number={1},
  
  pages={41--53},
  
  year={2024},
  
  publisher={Wiley Online Library}
  
}

@article{valderrama2024scientific,
  title={Scientific machine learning for predicting plasma concentrations in anti-cancer therapy},
  
  author={Valderrama, Diego and Teplytska, Olga and Koltermann, Luca Marie and Trunz, Elena and Schmulenson, Eduard and Fritsch, Schim and Jaehde, Ulrich and Froehlich, Holger},
  
  journal={medRxiv},
  
  pages={2024--05},
  
  year={2024},
  
  publisher={Cold Spring Harbor Laboratory Press}
}

To download everything from this repository onto your local directory, execute the following line on your terminal:

--------
## Contact
- [Prof. Dr. Holger Fröhlich](mailto:holger.froehlich@scai.fraunhofer.de)
- [Diego Valderrama](mailto:diego.felipe.valderrama.nino@scai.fraunhofer.de)
- AI and Data Science Group, Bioinformatics Department, Fraunhofer Institute for Algorithms and Scientific Computing (SCAI), Schloss Birlinghoven, 1, 53757 Sankt Augustin.
