# Assignment 1, Deep Learning Fundamentals, 2020

Python implementation of a single layer perceptron and a multi later perceptron from scrach, and comparison with Support Vector MAchine and Random Forest algorithms

Testing using the PIMA Indians Diabetes Dataset [link to Kaggle] (https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## Environment

This repo was tested under a Linux 64 bit OS, using Python 3.8.5

## How to run this repo

In order to use this repo:

1. Clone or download this repo

```bash
git clone https://github.com/juliancabezas/deep_learning_perceptron.git
```

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual)
3. Create a environment using the perceptron.yml file included in this repo, using the following command (inside conda or bash)

```bash
conda env create -f perceptron.yml --name perceptron
```

4. Activate the conda environment

```bash
conda activate perceptron
```

5. Run each specific file in yout IDE of preference, (I recommend [VS Code](https://code.visualstudio.com/) with the Python extension), using the root folder of the directory as working directory to make the relative paths work.

Each file contains the workflow for each algorithm (Perceptron, Multilayer Perceptron, Random forest, Support Vector Machine)

* Alternatevely, you can build your own environment following the package version contained in requeriments.tx