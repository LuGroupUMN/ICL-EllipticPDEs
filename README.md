## Introduction

This repository contains the numerical implementation of the Transformer architecture introduced in the paper *"Provable In-Context Learning of Linear Systems and Linear Elliptic PDEs with Transformers."*

The repository is organized as follows:

### Directory Structure:

- **`src/`**: Contains the core source code for the project:
  
  - `data_generator.py`: Responsible for generating data for training and testing.
  
  - `models.py`: Defines the model architecture and the training paradigm used in this project.
  
  - `utils.py`: Includes utility functions, such as metric calculations and other basic functions.

- **`examples/`**: This directory contains examples to help users better understand the code. These examples correspond to figures presented in the paper. Please note that no data is included in this repository. Users will need to generate and train the Transformer themselves. The examples include:

  - **`Transformer_ICL_example.ipynb`**: A general workflow demonstrating how to generate data, train a simple Transformer as described in the paper, and use the trained model for testing.

  - **`Fig1_Scaling_Law/`**: Demonstrates how to generate data for scaling law analysis with respect to parameter "N" (the number of tasks used for training). This folder also contains the workflow for plotting the scaling law results, corresponding to Fig. 1 (A) in the paper.

  - **`Fig2_PDE_Error/`**: Provides an example of: 1) Analyzing the error of solving real Partial Differential Equations (PDEs) using the Transformer model. 2) Plotting the PDE solution and comparing it with numerical solutions from the Galerkin method. These visualizations correspond to Fig. 2 in the paper.

  - **`Fig3_Task_Shift_and_Cov_Shift/`**: Shows how to reproduce the task shift and covariance shift analysis discussed in the paper (Fig. 3). This folder provides instructions for generating these shift results and reproducing Fig. 3 (B).

## Requirements

To run this project, you need to install the following dependencies:

- `torch`
- `seaborn`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`


## Code Version

The code version is **v1.0.0**, released on **2024/10/15**.

## Code Authors

- **Tianhao Zhang** ([zhan7594@umn.edu](mailto:zhan7594@umn.edu)) – Developed the algorithms and implemented the codebase.
- **Riley O'Neill** ([oneil571@umn.edu](mailto:oneil571@umn.edu)) – Contributed to algorithm design and assisted in testing.
- **Frank Cole** ([cole0932@umn.edu](mailto:cole0932@umn.edu)) and **Yulong Lu** ([yulonglu@umn.edu](mailto:yulonglu@umn.edu)) – Provided discussion and feedback on the algorithm and experimental design.



## Citation

If you use this code, please cite the following paper:

> Cole F, Lu Y, O'Neill R, Zhang T. *Provable In-Context Learning of Linear Systems and Linear Elliptic PDEs with Transformers.* arXiv preprint arXiv:2409.12293. 2024 Sep 18.




