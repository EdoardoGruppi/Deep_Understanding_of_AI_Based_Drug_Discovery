# Import packages
from rdkit.Chem import Descriptors, AllChem as Chem, DataStructs
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Lipinski import NumRotatableBonds, NumHDonors, NumHAcceptors
from rdkit.Chem.MolSurf import TPSA
import numpy as np
import os
import joblib
import pickle
from Datasets.selected_targets import o_dict, pn_dict, th_dict
from pandas import DataFrame

# IMPORTANT: the model used is called ChEMBL_27 and is published by the ChEMBL group in one of their official
# repository called "of_conformal". The link to access the repository is: https://github.com/chembl/of_conformal. The
# paper describing the model is: https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-018-0325-4.pdf.

# The authors suggested to work with the official code using docker to build the environment. In this case instead the
# environment, whose packages can be found in the requirements.txt file, has been directly created with conda.
# The code is also slightly modified to work directly in local without the need to send https requests and
# consequently to handle the responses. Following this strategy, errors due to timeout are avoided as well. Finally,
# comments and definitions are also added to facilitate the comprehension of the users.

# Number of bits of the Morgan fingerprints used
N_BITS = 1024
# Name of the directory hosting the model files
INPUT_DIR = os.path.join("./Models", "chembl_mcp_models")

models = {}
scalers = {}
# Load all the files within the folder downloaded
for target_id in pn_dict.keys():
    # Load models
    model_path = f"{INPUT_DIR}/models/{target_id}/{target_id}_conformal_prediction_model"
    models[target_id] = joblib.load(model_path)
    # Load scalers
    scaler_path = f"{INPUT_DIR}/scalers/{target_id}_scaler.pkl"
    scalers[target_id] = pickle.load(open(scaler_path, "rb"))


def pred_category(p0, p1, significance):
    """
    Assign the molecule activity behaviour to a specific class according to the p-values obtained. Molecules can be
    assigned to the following classes: 'active', 'inactive', 'both' or 'empty'.

    :param p0: first p-value obtained from the model.
    :param p1: second p-value obtained from the model.
    :param significance: significance level to overcome to be assigned to a specific class. It is associated with the
        level of confidence of the prediction.
    :return: the predicted class.
    """
    if (p0 >= significance) & (p1 >= significance):
        return "both"
    if (p0 >= significance) & (p1 < significance):
        return "inactive"
    if (p0 < significance) & (p1 >= significance):
        return "active"
    else:
        return "empty"


def predict(descriptors):
    """
    Predict the activity behaviour of a molecule towards a set of 500 different targets.

    :param descriptors: molecule information and descriptors.
    :return: the predicted activity of the molecule towards each target.
    """
    # Empty list where to save the results
    predictions = []
    # For every target
    for target in models:
        # Load the corresponding scalers
        scaler = scalers[target]
        # Transform the input as required by the model
        X = np.column_stack((scaler.transform(np.array(descriptors[:6]).reshape(1, -1)),
                             descriptors[-1].reshape(1, -1),))
        # Predict the activity values
        pred = models[target].predict(X)
        # Get the P values
        p0 = float(pred[:, 0])
        p1 = float(pred[:, 1])
        # Format output for a single prediction
        res = {"Target_chembl_id": target,
               "Organism": o_dict[target],
               "Pref_name": pn_dict[target],
               "70%": pred_category(p0, p1, 0.3),
               "80%": pred_category(p0, p1, 0.2),
               "90%": pred_category(p0, p1, 0.1),
               "Threshold": th_dict[target]}
        # Append the results to the dedicated list
        predictions.append(res)
    return predictions


def calc_descriptors(rd_mol):
    """
    Compute the Morgan fingerprints and the molecule information required as input by the model.

    :param rd_mol: rd mol object.
    :return: a list of all the descriptors information.
    """
    # Get Morgan fingerprint
    fp = Chem.GetMorganFingerprintAsBitVect(rd_mol, radius=2, nBits=N_BITS, useFeatures=False)
    np_fp = np.zeros(N_BITS)
    # Convert the fingerprint into a numpy array
    DataStructs.ConvertToNumpyArray(fp, np_fp)
    # Get the octanol-water partition coefficient
    log_p = MolLogP(rd_mol)
    #  Get molecular weight
    mwt = Descriptors.MolWt(rd_mol)
    # Get number of rotatable bonds
    rtb = NumRotatableBonds(rd_mol)
    # Get number of hydrogen bond donors
    hbd = NumHDonors(rd_mol)
    # Get number of hydrogen bond acceptors
    hba = NumHAcceptors(rd_mol)
    # Get the topological polar surface area
    tpsa = TPSA(rd_mol)
    return [log_p, mwt, rtb, hbd, hba, tpsa, np_fp]


def handle_normal(smiles):
    """
    Computes the activity of a given molecule.

    :param smiles: smiles string defining the molecule to evaluate.
    :return: a pandas dataframe with activity values.
    """
    predictions = []
    # Load molecule from smiles and calculate fp
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Compute descriptors
        descriptors = calc_descriptors(mol)
        # Predict the activity values
        predictions = predict(descriptors)
    # Return the results as a pandas dataframe
    predictions = DataFrame(predictions)
    return predictions
