import numpy as np 
import time
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

from utils.io_utils import save_models
from utils.dataset_builder import ConcatData


def MLP_CLS(patient_num, data_dir, save_dir):

    start = time.time() 
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(patient_num)):
        print(f"MLP Training fold {fold + 1}/{5}...")
        train_patients, test_patients = patient_num[train_idx], patient_num[test_idx]
        train_path = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if int(name.replace('.csv','').split("_")[1]) in train_patients]
        test_path = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if int(name.replace('.csv','').split("_")[1]) in test_patients]

        train_data = ConcatData(train_path)
        test_data = ConcatData(test_path)

        # seperate the features and labels
        features_train = train_data.drop(['seizure','ID'], axis=1).values
        target_train = train_data.seizure.values
        X_test = test_data.drop(['seizure','ID'], axis=1).values
        Y_test = test_data.seizure.values

        #SMOTE upsampling the minority class(seizures)
        sm = SMOTE(sampling_strategy='minority',random_state=42)
        X_train, y_train = sm.fit_resample(features_train, target_train)

        # Parameter distribution for MLPClassifier
        clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Approximate (16*4, 16*2) hidden layers
        activation='relu',  
        solver='adam',
        alpha=0.001,  # L2 Regularization (simulates weight decay)
        batch_size=1024,  # batch sizes 
        learning_rate_init=0.0005,  # Smaller learning rate for stability
        max_iter=150,  # More iterations for convergence
        random_state=42,
        early_stopping=True  # Stops training if no improvement
        )

        clf.fit(X_train, np.ravel(y_train))

        # Compute total parameters
        num_params = sum(W.size for W in clf.coefs_) + sum(b.size for b in clf.intercepts_)
        print("Total number of parameters:", num_params)

        probabilities = clf.predict_proba(X_test)          # the probability of predictions    
        class_predictions = clf.predict(X_test)             # predictions based on grades

        #save predictions on a column at the end of the feature csv file
        test_data['pred-lim1'] = probabilities[:,0]
        test_data['pred-lim2'] = probabilities[:,1]
        test_data['prediction'] = class_predictions

        # save prediction of test data
        test_data.to_csv(os.path.join(save_dir, f'test_prediction{fold}.csv'), index=False)

        #save the model
        save_models(clf, os.path.join(save_dir, f'model{fold}.sav'))

    end = time.time()
    total_time = end - start
    print(f'Total Execution Time: {total_time}') 

