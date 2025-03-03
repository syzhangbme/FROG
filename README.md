# FROG: A Fine-grained Spatiotemporal Graph Neural Network with Self-supervised Guidance for Early Diagnosis of Alzheimer’s Disease

by: Shuoyan Zhang, Jiehui Jiang, et al.

This work is currently under review by the IEEE Journal of Biomedical and Health Informatics.

## Requirments

python==3.10.13

pytorch==2.1.0

numpy==1.26.4

scipy==1.13.1

scikit-learn==1.5.0

pyyaml==6.0.1

## Preparing Data

The fMRI processed by DPABI contains the BOLD signals of 90 brain regions, and then undergoes successive processing steps such as format conversion, graph construction, normalization, padding, and augmentation. As indicated in the 'DataPreprocessing' folder.

## Model

The FROG model is in the Model_GNN.py file, where the *Autoencoder* class is used for pretraining and the *Funetune class* is used for finetuning.

## Pretraining FROG

```shell
python TrainTest_FROG_Autoencoder.py --configFile='./setting/FROG_Autoencoder_CoRR.yaml'
```

## Finetuning FROG

```shell
python TrainTest_FROG_Finetune.py --configFile='./setting/FROG_Finetune_CN_MCI_ALL.yaml'
```

## Configurations

Each pretraining and finetuning scenario is organized in the 'setting' folder, including using CoRR for pretraining, classification of CN vs. MCI, etc. It can be called in the shell command.
