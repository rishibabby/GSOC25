# GSOC25

## Project Title

Foundation Model for Gravitational Lensing

## Project Description

Strong gravitational lensing is a powerful tool for studying dark matter and the large-scale structure of the universe. This project focuses on developing a vision foundation model specifically designed for lensing data, which can be fine-tuned for a variety of downstream tasks, including classification, super-resolution, regression, and lens finding.

This project will explore different training strategies such as self-supervised learning, contrastive learning, or transformer-based models to learn meaningful representations of lensing images. By leveraging diverse datasets and training methodologies, the model will serve as a general-purpose backbone that can adapt to different astrophysical tasks while improving generalization across various observational conditions.


## Approach

To explore the most effective self-supervised learning approach for our task, we aim to train a range of self-supervised techniques, including Masked Autoencoders (MAE), SimCLR, MoCo, DINO, iBOT, and IJEPA, on the Model 2 dataset. Following the pretraining phase, we will fine-tune each model for classification using datasets from Model 1 and Model 3. This structured approach will allow us to systematically evaluate the performance of different self-supervised methods across varying datasets and tasks, providing insights into their generalization capabilities. The results from this comparative study will be documented in a structured table, highlighting key performance metrics for each technique.

Based on the findings, we will then extend our investigation to incorporate domain-specific knowledge through a physics-informed version of IJEPA. Given the nature of our problem, integrating physical constraints or priors into the self-supervised learning framework may enhance feature representations and improve downstream classification performance. 

## Results

Before training, Download the dataset from [repository](https://github.com/mwt5345/DeepLenseSim/tree/main/) and keep it in data folder. 

As first experiment, I have trained MAE on Model 2 dataset with 50k samples and finetuned classification layer for Model 3 dataset and test accuracy is shown in table below. I will further train other self supervised techniques. Finally, I want to build foundation model based on gravitaion lensing physics equations. 

| Method   | Pretraining Dataset | Fine-tuning Dataset | Accuracy (%) |
|----------|--------------------|---------------------|-------------|
| MAE      | Model 2            | Model 3            | 67%         |
| SIMCLR   | Model 2            | Model 3            | TBD         |
| MOCO     | Model 2            | Model 3            | TBD         |
| DINO     | Model 2            | Model 3            | TBD         |
| iBOT     | Model 2            | Model 3            | TBD         |
| IJEPA    | Model 2            | Model 3            | TBD         |
| **Ours**    | Model 2            | Model 3            | TBD         |

## Installation and Usage


```sh

git clone https://github.com/rishibabby/GSOC25.git
cd GSOC25
pip3 install -r requirements.txt
python3 -m scripts.run_mae
python3 -m scripts.finetune_mae

```

## Acknowledgment

I would like to thank my mentors, Pranath Reddy, Dr. Michael Toomey and Prof. Sergei Gleyzer, for their continued support and guidance.

