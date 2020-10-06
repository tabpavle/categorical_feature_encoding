from sklearn import preprocessing
from sklearn import ensemble
import os
import pandas as pd
from sklearn import metrics
import os
from . import dispatcher
import joblib



TRAINING_DATA = os.environ.get('TRAINING_DATA') # ENVIRONMENT VARIABLE
FOLD = int(os.environ.get('FOLD')) # ENVIRONMENT VARIABLE
MODEL = os.environ.get('MODEL') # ENVIRONMENT VARIABLE FROM DISPATCHER
TEST_DATA = os.environ.get('TEST_DATA') # ENVIRONMENT VARIABLE

# WE USE THIS TO SPLIT OUR DATA INTO TRAIN AND VALID DF
FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}



if __name__ == '__main__':

    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))] #FOR EXAMPLE If FOLD=0 THEN THAT IS OUR VALIDATION SET, REST IS FOR TRANING
    valid_df = df[df.kfold == FOLD]

    #WE DEFINE OUR TARGET VARIABLE IN BOTH SETS

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    #WE DROP ID, KFOLD AND TARGET FROM TRAINING AND VALID (we dont need id and kfold for running models and we previosly defined target variable)

    train_df = train_df.drop(['id', 'kfold', 'target'], axis=1)
    valid_df = valid_df.drop(['id', 'kfold', 'target'], axis=1)


    valid_df = valid_df[train_df.columns] # MAKE SURE THAT THERE ARE SAME COLUMNS WITH SAME ORDER


    # WE USE LABEL ENCODER FOR ENCODING CATEOGRICAL VARIABLES TO RIGHT FORMAT FOR TRAINING
    
    label_encoders = {}

    for c in train_df.columns: # WE GO TROUGHT EVERY COLUMNS
        print(c)
        lbl = preprocessing.LabelEncoder() # initiate LBL
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist()) # FIT ON OUR TRAIN AND VALID SAMPLES, AND ALSO HAS TO BE 1D ARRAY
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist()) # TRANSFORM TRAIN
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist()) # TRANSFORM VALID
        label_encoders[c] = lbl

    # DATA IS READY TO TRAIN



    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain) # FIT ON OUR TRAINING SET
    preds = clf.predict_proba(valid_df)[:, 1] # PREDICT PROBABILITIES OF BEINING 1 ON OUR VALID SET
    print(preds)
    print(metrics.roc_auc_score(yvalid, preds)) # CHECK OUR SCORE WITH ROC

    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl") # SAVE OUR LABEL ENCODERS 
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl") # SAVE OUR MODEL
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl") # SAVE OUR COLUMNS



    

