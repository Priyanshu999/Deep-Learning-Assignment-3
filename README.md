[![Project Report](https://img.shields.io/badge/Project%20Report-Click%20Here-blue?style=for-the-badge)](https://wandb.ai/cs24m035-indian-institute-of-technology-madras/CS6910_Assignment2_final2/reports/DA6401-Assignment-2---VmlldzoxMjM2MTg1Nw?accessToken=069sem3p535rxsxdu7t6j3lzj1njtbgxtcxgss3xrusg0amhip5thk4fnwh38ucn)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Priyanshu999/Deep-Learning-Assignment-3)

# Project Repository 📂

Welcome to this repository! 

## Table of Contents  
1. [Project Title & Description](#project-title--description)
2. [Project Structure](#project-structure)
3. [Features](#features)  
4. [Installation & Dependencies](#installation--dependencies)  
5. [Usage](#usage)  
6. [Acknowledgments](#acknowledgements)
    

# Encoder Decoder Model

## Project Title & Description  

In this assignment, we aim to explore sequence-to-sequence learning using various types of Recurrent Neural Networks (RNNs). The tasks involve designing and evaluating models based on **vanilla RNNs**, **LSTMs**, and **GRUs**, and investigating the role of **attention mechanisms** in enhancing sequence modeling. Additionally, we examine and visualize how different components of RNN-based architectures interact with each other during training and inference.

---

### Objectives

#### 1. Modeling Sequence-to-Sequence Learning Problems

The first task involves modeling sequence-to-sequence learning problems using Recurrent Neural Networks (RNNs).  
We aim to understand how RNN architectures can be applied to tasks where both input and output are sequences, such as:

- Machine translation  
- Text generation  
- Sequence prediction  

---

#### 2. Comparative Study of RNN Cells

This section involves comparing different RNN cell types:

- **Vanilla RNNs**
- **Long Short-Term Memory networks (LSTMs)**
- **Gated Recurrent Units (GRUs)**

The comparison focuses on:

- Performance  
- Training dynamics  
- Ability to capture long-term dependencies  

---

#### 3. Incorporating Attention Mechanisms

Here, we incorporate **attention mechanisms** into the seq2seq framework to:

- Help the model focus on relevant parts of the input sequence  
- Mitigate limitations of traditional RNN-based architectures  
- Improve performance on longer and more complex sequences  

---


## Project Structure

The root directory of this project contains the following components:

### 🧠 Notebooks and Python Scripts

- `vanilla_seq2seq.ipynb` — Jupyter notebook implementing sequence-to-sequence learning using a vanilla RNN.
- `attention_seq2seq.ipynb` — Jupyter notebook implementing sequence-to-sequence learning with attention mechanism.
- `vanilla_seq2seq.py` — Python script version of the vanilla RNN model.
- `attention_seq2seq.py` — Python script version of the attention-based model.

### 📂 Output Folders

- `vanilla_predictions/`  
  └── `output.csv` — Contains predicted sequences for the test set using the vanilla RNN model.

- `attention_predictions/`  
  └── `output.csv` — Contains predicted sequences for the test set using the attention-based model.

---

### 📌 Requirements

Make sure you have the required Python packages installed. 




## Features  

This project implements a fully customizable **Convolutional Neural network** with various tunable hyperparameters and optimization techniques.  

### 🔹 Supported Functionalities  
- Embedding dimension: {16, 32, 64, 256}
- Number of encoder layers: {1, 2, 3}
- Number of decoder layers: {1, 2, 3}
- Hidden size (RNN units): {16, 32, 64, 256}
- RNN cell type: {Vanilla RNN, GRU, LSTM}
- Dropout rate: {0.2, 0.3}
- Beam size (used during decoding): {1 (greedy), 3, 5}


### 🔹 Hyperparameter Tuning  
To efficiently search for optimal hyperparameters, we utilized **Weights & Biases (WandB) Sweep functionality**, allowing automated experimentation with different configurations for improved performance.  

This flexibility makes the model **highly configurable and scalable**, enabling experimentation with various architectures and optimization strategies! 🚀  

## Installation & Dependencies  

### 🔹 Prerequisites  
Ensure you have **Python 3.x** installed on your system. You can check your Python version using:  

```bash
python --version
```
or  
```bash
python3 --version
```

### 🔹 Cloning the Repository  
To download this project, open a terminal and run:  

```bash
git clone https://github.com/Priyanshu999/Deep-Learning-Assignment-3.git
cd Deep-Learning-Assignment-3
```

### 🔹 Installing Dependencies  
Install all necessary dependencies first.

After installation, you're all set to run the project! 🚀


## Usage

### 🔹 Running Training Script
You can run either of the Python scripts (`vanilla_seq2seq.py` or `attention_seq2seq.py`) using the command line with the following arguments:

| Argument | Long Form | Description | Type | Default | Choices |
|----------|-----------|-------------|------|---------|---------|
| `-wp` | `--wandb_project` | Project name used to track experiments in Weights & Biases dashboard | `str` | `'DL-Assignment3'` | — |
| `-we` | `--wandb_entity` | Wandb Entity used for experiment tracking | `str` | `'cs23m026'` | — |
| `-d` | `--datapath` | Path to the dataset | `str` | `'D:/DL_A3/Dataset'` | — |
| `-l` | `--lang` | Language code | `str` | `'hin'` | — |
| `-e` | `--epochs` | Number of training epochs | `int` | `10` | — |
| `-b` | `--batch_size` | Batch size | `int` | `32` | — |
| `-dp` | `--dropout` | Dropout probability in encoder & decoder | `float` | `0.3` | — |
| `-nl` | `--num_layers` | Number of layers in encoder & decoder | `int` | `2` | — |
| `-bw` | `--beam_width` | Beam width for beam search | `int` | `1` | — |
| `-cell` | `--cell_type` | RNN cell type for encoder & decoder | `str` | `'LSTM'` | `['LSTM', 'RNN', 'GRU']` |
| `-emb_size` | `--embadding_size` | Embedding size | `int` | `256` | — |
| `-hdn_size` | `--hidden_size` | Hidden layer size | `int` | `512` | — |
| `-lp` | `--length_penalty` | Length penalty for beam search | `float` | `0.6` | — |
| `-bi_dir` | `--bidirectional` | Use bidirectional encoder (1 = yes, 0 = no) | `int` | `1` | `[0, 1]` |
| `-tfr` | `--teacher_forcing_ratio` | Teacher forcing ratio | `float` | `0.5` | — |
| `-o` | `--optimizer` | Optimizer type | `str` | `'adam'` | `['sgd', 'rmsprop', 'adam', 'adagrad']` |
| `-lr` | `--learning_rate` | Learning rate | `float` | `0.001` | — |
| `-p` | `--console` | Print training/validation metrics every epoch | `int` | `1` | `[0, 1]` |
| `-wl` | `--wandb_log` | Enable Weights & Biases logging | `int` | `0` | `[0, 1]` |
| `-eval` | `--evaluate` | Run evaluation on test set | `int` | `1` | `[0, 1]` |
| `-t_random` | `--translate_random` | Print 10 random test translations | `int` | `0` | `[0, 1]` |



## Acknowledgements
I would like to acknowledge the following resources that helped shape this project:

- [Mitesh Khapra Deep Learning Course](https://www.youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM)
- [Deep Learning - Stanford CS231N](https://www.youtube.com/playlist?list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ)


🔗 Happy Coding! 😊
