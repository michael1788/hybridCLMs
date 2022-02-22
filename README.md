# Leveraging molecular structure and bioactivity with chemical language models for drug design

## Table of Contents
1. [Description](#Description)
2. [Requirements](#Requirements)
3. [How to run an experiment](#Run)
4. [How to cite this work](#Cite)
5. [License](#license)
6. [Address](#Address)


### Description<a name="Description"></a>

Supporting code for the paper «Leveraging molecular structure and bioactivity with chemical language models for drug design»   

[Preprint version](https://chemrxiv.org/engage/chemrxiv/article-details/615580ced1fc334326f9356e)   

**Abstract of the paper**: Generative chemical language models (CLMs) can be used for de novo molecular structure generation. These CLMs learn from the structural information of known molecules to generate new ones. In this paper, we show that “hybrid” CLMs can additionally leverage the bioactivity information available for the training compounds. To computationally design ligands of phosphoinositide 3-kinase gamma (PI3Kγ), we created a large collection of virtual molecules with a generative CLM. This primary virtual compound library was further refined using a CLM-based classifier for bioactivity prediction. This second hybrid CLM was pretrained with patented molecular structures and fine-tuned with known PI3Kγ binders and non-binders by transfer learning. Several of the computer-generated molecular designs were commercially available, which allowed for fast prescreening and preliminary experimental validation. A new PI3Kγ ligand with sub-micromolar activity was identified. The results positively advocate hybrid CLMs for virtual compound screening and activity-focused molecular design in low-data situations.


### Requirements<a name="Requirements"></a>

First, you need to [clone the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository):

```
git clone git@github.com:michael1788/hybridCLMs.git
```
Then, you can run the following command, which will create a conda virtual environment and install all the needed packages (if you don't have conda, you can follow the instructions to install it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

```
cd hybridCLMs/
conda env create -f environment.yml
```

Once the installation is done, you can activate the virtual conda environment for this project:

```
conda activate hybrid
```
Please note that you will need to activate this virtual conda environment every time you want to use this project. 

### How to run an experiment<a name="Run"></a>

You can run an example experiment based on the data used in the paper by following the procedure described in A and B.

Note: we provide the pretrained weights for both the [CLM](https://github.com/michael1788/hybridCLMs/tree/main/pretrained_models) and the [E-CLM](https://github.com/michael1788/hybridCLMs/tree/main/pretrained_models). You can therefore skip steps A1, A2, and B1 if you do not want to repeat the whole experiment, or if you do not have access to at least one GPU.

A. Generate the focused chemical library

We start by generating the focused virtual chemical library in part A.

A1. Process the data to train the chemical language model (CLM):
```
cd experiments/
sh run_processing.sh configfiles/clm/A01_clm.ini
```
Note: you can skip this step, as we provide the [processed pretraining data](https://github.com/michael1788/hybridCLMs/tree/main/data/us_pharma_patent_data_lowe_smiles_can_unique_stereochem/1_90_x0).

A2. Pretrain the CLM:
```
sh run_training.sh configfiles/clm/A01_clm.ini
```
Note: we encourage you to skip this step, and to use the available pretrained model, especially if you do not have a GPU.

A3. Process the data to fine-tune the CLM:
```
sh run_processing.sh configfiles/ft_clm_generation/A01_clm_ft.ini
```

A4. Fine-tune the pretrained CLM:
```
sh run_training.sh configfiles/ft_clm_generation/A01_clm_ft.ini
```

A5. Generate SMILES strings with the fine-tuned CLM:
```
sh run_generation.sh configfiles/ft_clm_generation/A01_clm_ft.ini
```
Note: this step will be slow if you sample 5,000 SMILES strings by epoch as specified in *A01_clm_ft.ini* without a GPU. We advised you to first try with 500 SMILES strings (to do so, you can update the value in *A01_clm_ft.ini*).

A6. Process the generated SMILES strings to get the new molecules to constitute the focused virtual chemical library:
```
sh run_novo.sh configfiles/ft_clm_generation/A01_clm_ft.ini
```

B. Refine the focused chemical library

In part B, we refine the focused virtual chemical library by leveraging the bioactivity data of the fine-tuning set.

B1. Pretrain the E-CLM:
```
sh run_training.sh configfiles/eclm/A01_eclm.ini
```
Note: we encourage you to skip this step, and to use the available pretrained model, especially if you do not have a GPU.

Next, we fine-tune the pretrained E-CLM to do the ordinal classification. We start with a cross-validation to find a suitable set of hyperparameters (e.g. the number of fine-tuning epochs).

B2. Run a cross-validation experiment:
```
sh run_experiment.sh configfiles/ft_eclm/A01_cv.ini
```

B3. Run the analysis of the cross-validation experiment:
```
sh run_analysis.sh configfiles/ft_eclm/A01_cv.ini
```
You will find in *hybridCLMs/experiments/outputs/ft_eclm/A01_cv/analysis/* a plot with your results, as well as a *.csv* file version of it. 

Note: you can explore hyperparameters by running commands B2 and B3 on other configuration files, where you can change the hyperparameters you want to explore (e.g. the learning rate or the number of fine-tuning epochs).

B4. Once you are satisfied with your cross-validation experiment(s), you can run the E-CLM once on the test set to assess the final performance of your hyperparameters. For example, for the hyperparameters we used in the cross-validation in B2 and B3:
```
sh run_experiment_test_set.sh configfiles/ft_eclm/A02_test.ini
```

B5. And run again the analysis:
```
sh run_analysis_test_set.sh configfiles/ft_eclm/A02_test.ini
```

B6. You can now train on all the data the best E-CLM, as defined by the results on the test set. This will create an ensemble of models (with the number of models specified in the configuration file), which will be used for deployment, i.e. to make predictions on the focused virtual chemical library: 

```
sh run_experiment_alldata.sh configfiles/ft_eclm/A03_alldata.ini
```

B7. Finally, we can use the deep ensemble of E-CLMs to refine the focused virtual chemical library generated in A.:

```
sh run_ensemble.sh configfiles/ensemble/A01_ensemble.ini
```
The results of the ensemble prediction can be found in a *.csv* file: *hybridCLMs/experiments/outputs/ensemble/A01_ensemble/*.

### How to cite this work<a name="Cite"></a>
```
@article{Moret2021,
  title={Leveraging molecular structure and bioactivity with chemical language models for drug design},
  author={Moret, Michael and Grisoni, Francesca and Brunner, Cyrill and Schneider, Gisbert},
  journal={Preprint at https://chemrxiv.org/engage/chemrxiv/article-details/615580ced1fc334326f9356e},
  year={2021},
}
```

### License<a name="License"></a>
[MIT License](LICENSE)


### Address<a name="Address"></a>
MODLAB   
ETH Zurich   
Inst. of Pharm. Sciences   
HCI H 413   
Vladimir-​Prelog-Weg 4   
CH-​8093 Zurich   
