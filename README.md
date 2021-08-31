# Description of the project

[Project](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery) ~ [Related Project](https://github.com/EdoardoGruppi/Drug_Design_Models)
 ~ [Guide](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery/blob/main/Instructions.md)

The present study is finalised to determine the most advanced models in the literature capable of producing new
high-quality molecules starting from well-known datasets. The selection is carried out through a series of evaluation
processes. At first, the output samples of each method are evaluated according to certain physico-chemical properties
such as Quantitative Estimation of Drug-likeness (QED) and Synthetic Accessibility (SA). Then, in a successive step, the
assessment also includes the predicted activity towards one target protein. The final aim of the project actually is to
better understand whether and how the performance of each model varies when the typology of the target protein is
changed. More precisely, the proteins involved in this work are:  Beta-secretase 1 (BACE1), Peroxisome
proliferator-activated receptor alpha (PPAR-α), Cyclin Dependent Kinase 2 (CDK2) and the Dopamine Receptor subtype D3 (
DRD3).

The modified code used to run the models is provided in the GitHub repo accessible at the
following [link](https://github.com/EdoardoGruppi/Drug_Design_Models). Specifically, the code at the provided link is a
slightly updated version of that published by the authors in their projects. In case an error is encountered, it means
the page has not been already published.

**Note:** the content of the folder Models/chembl_mcp_models is retrieved from the official
[GitHub page](https://github.com/chembl/of_conformal) of the ChEMBL group.

## Content

This project includes a series of experiments conducted at the beginning of the learning phase with the objective to
acquire a practical knowledge of the inner workings of the models aimed to produce molecules.

Nevertheless, the main contribution is
the [test_script.py](https://github.com/EdoardoGruppi/Deep_Understanding_of_AI_Based_Drug_Discovery/blob/main/test_script.py)
file that enables to have a comprehensive overview of the quality of the molecules generated. The benchmark exploited by
the script is created adding some brand-new metrics to those presented in
the [GuacaMol](https://github.com/BenevolentAI/guacamol) and [MOSES](https://github.com/molecularsets/moses)
benchmarking platforms.

## How to start

A comprehensive guide concerning how to run the code along with additional information is provided in the
file [Instruction.md](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery/blob/main/Instructions.md)
.

The packages required for the execution of the code along with the role of each file and the software used are described
in the Sections below.

## Packages required

Althoug the following list gather all the most important packages needed to run the project code, a more comprehensive
overview is provided in the
file [requirements.txt](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery/blob/main/requirements.txt)
. The latter can also be directly used to install the packages by typing the specific command on the terminal. Please
note that the descriptions provided in this subsection are taken directly from the package source pages. For more
details it is reccomended to directly reference to the related official websites.

**Compulsory :**

- **Pandas** provides fast, flexible, and expressive data structures designed to make working with structured and time
  series data both easy and intuitive.

- **Numpy** is the fundamental package for array computing with Python.

- **Tensorflow** is an open source software library for high performance numerical computation. Its allows easy
  deployment of computation across a variety of platforms (CPUs, GPUs, TPUs). **Important**: Recently Keras has been
  completely wrapped within Tensorflow.

- **Os** provides a portable way of using operating system dependent functionality.

- **Matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.

## Role of each file

**main.py** it is the starting point of all the experiments conducted to understand the mechanisms of the de novo drug design generative models.

**test_script.py** contains a wide set of metrics useful to understand the quality of a collection of molecules.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the
most advanced working environments and for its ease of use.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular
machine learning libraries and it offers GPUs where you can execute the code as well.

> <img src="https://user-images.githubusercontent.com/674621/71187801-14e60a80-2280-11ea-94c9-e56576f76baf.png" width="80" alt="vscode">

Visual Studio Code is a code editor optimized for building and debugging modern web and cloud applications.
