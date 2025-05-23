import argparse
import glob
import os
import random
import sys
from datetime import date

import numpy as np

from evaluation import Eval_folds
from models.MLP import MLP_CLS
from models.RF import RF_CLS
from models.XGBoostmodel import XGBoost_CLS
from utils.io_utils import folder_creator

today = date.today()

# paths
PYTHON = sys.executable
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    default="path to your files",
    help="Directory containing data of features",
)

parser.add_argument(
    "--save_dir", 
    default=os.path.join("results", str(today)), 
    help="Directory for saving the result"
)

args = parser.parse_args()

# Make a folder to save files
folder_creator(args.save_dir)

# Set random seeds
np.random.seed(42)
random.seed(42)

# Get the patient numbers
patient_number = np.array([int(name.replace('.csv','').split("_")[1]) for name in os.listdir(args.data_dir)])

# Subdirectories for each model
mlp_dir = os.path.join(args.save_dir, "MLP")
rf_dir = os.path.join(args.save_dir, "RF")
xgboost_dir = os.path.join(args.save_dir, "XGBoost")

# MLP classification and evaluation
MLP_CLS(patient_number, args.data_dir, mlp_dir)
Eval_folds(mlp_dir, classifier_name="MLP")

# Random Forest classification and evaluation
RF_CLS(patient_number, args.data_dir, rf_dir)
Eval_folds(rf_dir, classifier_name="RF")

# XGBoost classification and evaluation
XGBoost_CLS(patient_number, args.data_dir, xgboost_dir)
Eval_folds(xgboost_dir, classifier_name="XGBoost")
