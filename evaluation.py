import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import pickle


def Eval_folds(save_dir, classifier_name):

    dataframes = [pd.read_csv(os.path.join(save_dir, f'test_prediction{i}.csv')) for i in range(5)]

    result_total = pd.DataFrame()

    for i, df_test in enumerate(dataframes):

        target = df_test['seizure'].values  #one baby seizure
        pred_prob = df_test['pred-lim2'].values
        pred= df_test['prediction'].values

        cm = confusion_matrix(target, pred)
        TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        sens = TP/(TP+FN)
        spc = TN/(TN+FP)
        print('sensitivity = ', round(sens*100,3), 'specificity = ', round(spc*100,3))

        # The Matthews correlation coefficient
        mcc = matthews_corrcoef(target, pred)
        print('mcc:', mcc)

        # ROC calculation
        fpr, tpr, thresholds  = roc_curve(target, pred_prob)    #compare annotaion file labels with classifier prediction result 
        sensitivity = tpr
        specificity = 1-fpr
        auc_value = metrics.auc(fpr, tpr)
        print('auc value:', round(auc_value*100,3))

        if classifier_name == 'RF' or classifier_name == 'XGBoost':
            loaded_model = pickle.load(open(f'{save_dir}/model{i}.sav', 'rb'))
            #check the importance of features in classification
            df_features_name=['muNN','SDNN','RMSSD','TRindex','kurtosis','skewness','ApEn','LF/HF ratio','VLF power','LF  power','HF power','SD1','SD2','CSI','CVI','SD ratio']  
            importances = loaded_model.feature_importances_
            #plot feature importance
            plt.figure()
            plt.bar(df_features_name, importances, color='purple')
            plt.xticks(rotation = 90)
            plt.ylabel("Importance value")
            plt.title('HIE Grading Fearure Importance')
            plt.savefig(f'{save_dir}/feature_importance_fold{i}.png', bbox_inches = 'tight')
            plt.close()


        df = pd.DataFrame({
        'TN': [TN],
        'FP': [FP],
        'FN': [FN],
        'TP': [TP],
        'sensitivity (%)': [round(sens * 100, 2)],
        'specificity (%)': [round(spc * 100, 2)],
        'mcc': [round(mcc, 2)],  # Usually not a percentage
        'auc': [round(auc_value * 100, 2)]
        })

        result_total = pd.concat([result_total, df], axis=0)
    result_total.to_csv(os.path.join(save_dir, 'total_evaluation_folds.csv'), index=False)
    AUC_folds_average = result_total['auc'].mean()
    print('AUC folds average:', AUC_folds_average)
    