from config import *
import pandas as pd

def get_predict(model, X_test):

    '''Model prediction (model, dataset)'''

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=['is_churned'])
    y_pred_df.to_csv(OUTPUT_DATA_PATH,  index=None)
    print(f'Prediction succesfully saved to {OUTPUT_DATA_PATH}')