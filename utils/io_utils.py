import pickle
import os


def save_models(model, save_dir):
    """save the model

    Args:
        model (_type_): model to save
        save_dir (string): directory to save the model
    """
    pickle.dump(model, open(save_dir, 'wb'))




def folder_creator(dir_path):
    """create a new folder

    Args:
        dir_path (string): path to make the new folder
    """
    project_names = ['MLP', 'RF', 'XGBoost']
    if not os.path.exists(dir_path):
        for name in project_names:
            os.makedirs(os.path.join(dir_path, name), exist_ok=True)
     