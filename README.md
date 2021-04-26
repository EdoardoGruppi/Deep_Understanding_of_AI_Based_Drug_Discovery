# Description of the project

[Project](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery) ~ [Guide]()

## How to start

A comprehensive guide concerning how to run the code along with additional information is provided in the file [Instruction.md]().

The packages required for the execution of the code along with the role of each file and the software used are described in the Sections below.

## Packages required

Althoug the following list gather all the most important packages needed to run the project code, a more comprehensive overview is provided in the file [requirements.txt](). The latter can also be directly used to install the packages by typing the specific command on the terminal.
Please note that the descriptions provided in this subsection are taken directly from the package source pages. For more details it is reccomended to directly reference to the related official websites.

**Compulsory :**

- **Pandas** provides fast, flexible, and expressive data structures designed to make working with structured and time series data both easy and intuitive.

- **Numpy** is the fundamental package for array computing with Python.

- **Tensorflow** is an open source software library for high performance numerical computation. Its allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs). **Important**: Recently Keras has been completely wrapped within Tensorflow.

- **Pathlib** offers a set of classes to handle filesystem paths.

- **Shutil** provides a number of high-level operations on files and collections of files. In particular, functions are provided which support file copying and removal.

- **Os** provides a portable way of using operating system dependent functionality.

- **Matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- **Seaborn** is a data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

- **Tqdm:** allows to display a smart progress meter to represent the status of a precise loop.

- **Sys** provides functions to manipulate different parts of the Python runtime environment.

## Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are realised. More precisely, it is responsible to call functions from other files in order to divide the datasets provided, pre-process images as well as to instantiate, train and test the models.

**config.py** makes available all the global variables used in the project.

**utilities.py** includes functions to download and split the datasets in the dedicated folder, to compute the mean RGB value of the dataset and to plot results.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the most advanced working environments and for its ease of use.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular machine learning libraries and it offers GPUs where you can execute the code as well.

> <img src="https://user-images.githubusercontent.com/674621/71187801-14e60a80-2280-11ea-94c9-e56576f76baf.png" width="80" alt="vscode">

Visual Studio Code is a code editor optimized for building and debugging modern web and cloud applications.

> <img src="https://camo.githubusercontent.com/9e56fd69605928b657fcc0996cebf32d5bb73c46/68747470733a2f2f7777772e636f6d65742e6d6c2f696d616765732f6c6f676f5f636f6d65745f6c696768742e706e67" width="140" alt="comet">

Comet is a cloud-based machine learning platform that allows data scientists to track, compare and analyse experiments and models.
