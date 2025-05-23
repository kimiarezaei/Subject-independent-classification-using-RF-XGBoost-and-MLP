import pandas as pd


def ConcatData(path_list):
    '''This function takes a list of file paths, reads each CSV file into a DataFrame,
    and concatenates them into a single DataFrame. It also drops any rows with missing values.'''
    dfs = []
    for file in path_list:
        df = pd.read_csv(file)
        df['ID'] = file.split('/')[-1].replace('.csv','')  # add the patient ID to the file
        dfs.append(df)    
    df_concat = pd.concat(dfs, ignore_index=True)  # Combine all DataFrames
    df_final = df_concat.dropna()  # Drop rows with missing values
    return df_final




        





           
 





