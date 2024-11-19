# AMSDT-Net: Slum Segmentation with Advanced Multi-Scale Dynamic Transformers Network
Advanced Multi-Scale Dynamic Transformers Network (AMSDT-Net) for high-precision slum area segmentation from satellite imagery. Includes architecture, training setup, datasets, and augmentation techniques.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Results](#results)
9. [Acknowledgements](#acknowledgements)

---

## Project Overview

AMSDT-Net is designed to segment slum areas from satellite images. The model integrates multi-scale feature extraction, dynamic transformers, and attention mechanisms to capture complex spatial details and improve boundary delineation. This repository includes the model code, dataset, and instructions to train and evaluate the model.

## Project Structure

Below is the file structure of the project:

```
AMSDT-Net-Slum-Segmentation
├── architecture
│   ├── init.py
│   ├── boundary_refine.py
│   ├── cbam.py
│   ├── dynamic_conv.py
│   ├── encoder.py
│   ├── fpn.py
│   ├── input.py
│   ├── loss.py
│   ├── model.py
│   ├── multiscale_feature.py
│   └── residual.py
├── configs
│   ├── config.yaml
│   ├── model.yaml
│   └── setup.yaml
├── config.yaml
├── dataset.py
├── initial_memo.txt
├── main.py
├── reference.txt
├── train.py
└── utils.py
```


---

## Dataset

The dataset consists of satellite images and corresponding segmentation masks:
- **unique_image**: RGB satellite images.
- **unique_mask**: Binary masks indicating slum areas.

Ensure that these folders are in the project’s root directory.

## Requirements

To run this project, you need the following:

- **Python** 3.8 or higher
- **PyTorch** (tested with version 1.8 or higher)
- Additional libraries listed in `requirements.txt`

---

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/msp99000/AMSDT-Net-Slum-Segmentation.git
    cd AMSDT-Net-Slum-Segmentation
    ```

2. **Install Dependencies**:
    Ensure you have Python installed, and then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

---

## Configuration

This project uses YAML files for configurations:

1. **Model Configuration** (`configs/model.yaml`): Contains parameters for the model’s architecture, such as input channels, transformer depth, and FPN channels.
2. **Training Configuration** (`configs/setup.yaml`): Defines training parameters, including batch size, learning rate, and early stopping.

Customize these configurations as needed before training or evaluation.

---

## Usage

### Step 1: Preparing the Dataset

Place the dataset folders (`unique_image` and `unique_mask`) in the root directory.

### Step 2: Training the Model

Use the `train.py` script to train the model:

```bash
python train.py
```

The script will use the configurations specified in configs/setup.yaml and save model checkpoints and logs in the specified directories.

### Step 3: Evaluating the Model

After training, you can evaluate the model using the `main.py` script:

```bash
python main.py
```

This script will load the latest model checkpoint and evaluate it on the validation set, saving the results and metrics in the log directory.

---

## Results
After running the evaluation script, you will find performance metrics and example results (such as accuracy, Intersection over Union (IoU), and boundary metrics) in the log directory. These metrics are essential for understanding the model's performance on the test data.

## Acknowledgements
This project was developed as part of a PhD research effort on automated slum segmentation. Special thanks to [Institution/University Name] for providing resources and support.
