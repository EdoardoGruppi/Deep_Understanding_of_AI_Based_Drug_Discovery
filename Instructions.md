# Instructions

## Setup

1. Install Tensorflow and all the other packages appointed in
   the [README.md](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery/blob/main/README.md)
   and [requirements.txt](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery/blob/main/requirements.txt)
   files.

2. Download the project directory
   from [GitHub](https://github.com/EdoardoGruppi/Graph_Based_Learning_For_Drug_Discovery).
3. Tensorflow enables to work directly on GPU without requiring explicity additional code. The only hardware requirement
   is having a Nvidia GPU card with Cuda enabled. To see if Tensorflow has detected a GPU on your device run the
   following few lines (see main.py).

   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   ```

   If not, there are lots of guides on the web to install everything you need. For instance, you can take a look at
   [this](https://deeplizard.com/learn/video/IubEtS2JAiY).

4. Finally, it is crucial to run the code below since Tensorflow tends to allocate directly all the GPU memory even if
   is not entirely needed. With these lines instead, it will allocate gradually the memory required by the program (see
   main.py).

   ```python
   if len(physical_devices) is not 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
   ```

5. The project is developed on tensorflow 2.4.1 and python 3.7. To work on GPUs you must install both Cuda 11.0 and
   cudnn 8.0.4. To install cuda version 11 run the following line on the project terminal:

   ```
   conda install -c anaconda cudatoolkit=11.0
   ```

   Then, download cudnn 8.0.4 from the nvidia website. Unzip the downloaded folder and move the folder named cuda in a
   selected location. Finally, add a new environment variable on windows as C:
   \whole-path-to-the-selected-folder\cuda\bin. Restart the pc.

   Alternatively, it is possible to run the code with lower versions of tensorflow and the disable_eager_mode activated.
   Nevertheless, several functions and methods including tf.GradientTape will not work properly.

6. To install Spektral only pip can be used. Hence, run the following line on the terminal:

   ```
   pip install spektral
   ```

7. The installation of rdkit needs an additional step. Firstly, run the following line on the terminal:

   ```
   conda install -c rdkit rdkit
   ```

   Then, if some problem occurs it could be necessary to reinstall the numpy version published previously with respect
   to the newest 1.20. Only in such circumstances, execute the following line:

   ```
   pip install numpy==1.19.5 --user
   ```

   **Important:** Finally, since some experimental functions not directly included in the package are used, it is
   necessary to copy and paste the Contrib folder from:

   ```
   C:\Users\<'name user'>\anaconda3\pkgs\<'rdkit version'>\Library\share\RDKit
   ```

   to:

   ```
   C:\Users\<'name user'>\anaconda3\envs\<'name env'>\Lib\site-packages\rdkit
   ```
   
   **Note:** alternatively, it is possible to download the Contrib folder from the official 
   [GitHub page](https://github.com/rdkit/rdkit/tree/master/Contrib) of rdkit. In particular, be sure that the downloaded 
   directory includes both the NP_Score and SA_Score subfolders.

8. To run the target prediction model is necessary to download the model and scalers files from the link provided by the authors of the ChEMBL_27 model (for more details visit their GitHub repository [chembl/of_conformal](https://github.com/chembl/of_conformal)). The files required can be downloaded by typing in the windows terminal under a precise location the following line:

   ```
   !curl -SSL ftp://ftp.ebi.ac.uk/pub/databases/chembl/target_predictions/MCP/chembl_27_mcp_models.tar.gz -o models.tar.gz
   ```

   Hence, unzip the tar.gz folder retrieved and insert the chembl_mcp_models folder in the Models directory.

9. Ensure that all the packages required by the target prediction model are already installed. For instance, run the following lines:
   ```
   pip install https://github.com/josecarlosgomezt/nonconformist/archive/master.zip
   pip install lightgbm==2.3.0
   pip install dill
   pip install scikit-learn==0.22
   ```

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal or by the
specific command within the IDE.

```
python main.py
```

## Issues

- If the device's GPU memory is not enough to run the code, it is possible to execute the training of each model inside
  a dedicated subprocess. Tensorflow then will release the part of GPU memory used by each subprocess as soon as it
  ends.
