import pandas as pd 
import os 
from sklearn import preprocessing 
from sklearn import ensemble 
from sklearn import model_selection
import joblib
import numpy as np 
from . import dispatcher 



MODEL = os.environ.get('MODEL')
TEST_DATA = os.environ.get('TEST_DATA')



def predict():
    df = pd.read_csv(TEST_DATA) # READ TEST DATA
    predictions = None
    test_idx = df['id'].values # FOR SUBMISSION LATER

    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA) # DATA IS CHANING FOR EVERY FOLD SO WE HAVE TO CALL IT AGAIN EVERY TIME

        encoders = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}_label_encoder.pkl")) # LOAD OUR DICT WITH COLUMNS AND ENCODERS

        for c in encoders: # FOR EVERY COLUMN  WE LOAD OUR FITTED ENCODERS
            lbl = encoders[c] # INITATE ENCODER FOR C COLUMN
            df.loc[:, c] = lbl.transform(df[c].values.tolist()) # TRANSFORM OUR TEST SET

        clf = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}.pkl")) # LOAD OUR MODELS
        cols = joblib.load(os.path.join('models', f"{MODEL}_{FOLD}_columns.pkl")) # LOAD OUR COLUMNS
        df = df[cols]
        preds = clf.predict_proba(df)[: , 1]
    
        if FOLD == 0:
            predictions = preds
        
        else:
            predictions += preds    

    predictions /= 5 # AVERAGE OF 5 FOLDS
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns= ['id', 'target'])

    return sub


if __name__ == '__main__':
    submission = predict()
    submission.to_csv('models/random_forest.csv', index=False)




