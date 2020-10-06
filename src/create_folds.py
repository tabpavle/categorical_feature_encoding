import pandas as pd
from sklearn import model_selection


#GENERATING DATAFRAME WITH KFOLD COLUMN FOR CROSS VALIDATION IN TRAINING

if __name__ == '__main__':
    # READ DATA
    df = pd.read_csv('input/train.csv')
    # MAKE A COLUMN KFOLD WITH FAKE VALUE -1
    df['kfold'] = -1
    # SHUFFLE DATA
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    # WE USE STRATIFIEDKFOLD SO TARGET DISTRIBUTION WOULD BE EQUALLY DISTRIBUTED IN CROSS VALIDATION 
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42) # WE ALREADY SHUFFLED IT  
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(fold)
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold # WE FILL UP KFOLD COLUMN WITH FOLD FROM 0 TO 4 (N_SPLITS=5)


    df.to_csv('models/train_folds.csv', index=False) # WE SAVE OUR DATAFRAME




