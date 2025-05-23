import numpy as np 
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE

from utils.io_utils import save_models
from utils.dataset_builder import ConcatData


def RF_CLS(patient_num, data_dir, save_dir):

    start = time.time() 
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_estimators = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(patient_num)):
        print(f"RF Training fold {fold + 1}/{5}...")
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

        param_grid = {
        'max_features': ['sqrt'],   # Maximum depth of the trees
        'max_depth': [5, 7, 10],    # Step size shrinkage
        'n_estimators': [100],  # Number of boosting rounds
        'min_samples_split': [5, 8, 10],     # Subsample ratio of the training instances
        'min_samples_leaf': [3, 5, 8],      # Subsample ratio of columns for each tree
        'criterion' : ['gini']  # Minimum sum of instance weight(hessian) needed in a child
        }

        clf = RandomForestClassifier(random_state=42)

        # Run randomized search
        random_search = RandomizedSearchCV(
        clf,
        param_distributions=param_grid,
        n_iter=10,               # Number of combinations 
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
        )
        random_search.fit(X_train, np.ravel(y_train))

        best_params = random_search.best_params_
        # Remove the old value from the grid search
        best_params.pop('n_estimators', None)
        best_clf = RandomForestClassifier(**best_params, n_estimators=1000, random_state=42)
        best_clf.fit(X_train, np.ravel(y_train))

        # Save or store final model
        best_estimators.append(best_clf) 

        probabilities = best_clf.predict_proba(X_test)          # the probability of predictions    
        class_predictions = best_clf.predict(X_test)             # predictions based on grades

        #save predictions on a column at the end of the feature csv file
        test_data['pred-lim1'] = probabilities[:,0]
        test_data['pred-lim2'] = probabilities[:,1]
        test_data['prediction'] = class_predictions

        # save prediction of test data
        test_data.to_csv(os.path.join(save_dir, f'test_prediction{fold}.csv'), index=False)

        #save the model
        save_models(best_clf, os.path.join(save_dir, f'model{fold}.sav'))

    end = time.time()
    total_time = end - start
    print(f'Total Execution Time: {total_time}') 

