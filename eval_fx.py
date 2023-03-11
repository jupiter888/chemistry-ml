import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

### Upcoming modules:
    # Initialize eval-results-df-appender, train-and-test-predictions-df-appender, model hyperparameter arrays
    # 
    # eval_df = pd.DataFrame(columns=['Dataset', 'ML Model', 'TRAIN MSE', 'TRAIN R2', 'TRAIN MAE', 'TEST MSE', 'TEST R2', 'TEST MAE', 'params_string'])
    # preds_df = pd.DataFrame(columns=['Dataset', 'ML Model', 'Train Prediction Results', 'Test Prediction Results', 'params_string'])
    # params_string = str(model.get_params())
    
    
##### Put eval_results under model.predict#####
def eval_results(model, data_set, model_name, Y_train, Y_test, Y_pred_train, Y_pred_test):

    # Evaluate the model's performance on the training data
    train_mse = mean_squared_error(Y_train, Y_pred_train)
    #rmse_train = mean_squared_error(Y_train, Y_pred_train, squared=False)
    train_r2 = r2_score(Y_train, Y_pred_train)
    train_mae = mean_absolute_error(Y_train, Y_pred_train)

    # Evaluate models performance on test data
    test_mse = mean_squared_error(Y_test, Y_pred_test)
    #rmse_test = mean_squared_error(Y_test, Y_pred_test, squared=False)
    test_r2 = r2_score(Y_test, Y_pred_test)
    test_mae = mean_absolute_error(Y_test, Y_pred_test)
    
    print(f"PREDICTIONS OF {model_name} MODEL ON {data_set} DATASET:")
    print(f"Train Predictions({data_set}):\n{Y_pred_train}\n-----------------------------------------\n")
    print(f"Test Predictions({data_set}):\n{Y_pred_test}\n--------------------------------------\n")
    
    print(f"{model_name} MODEL EVALUATION FOR {data_set} DATASET:")
    print(f"Train MSE: {train_mse} Train R2: {train_r2} Train MAE: {train_mae}\nTest MSE: {test_mse} Test R2: {test_r2} Test MAE: {test_mae}")
    print(f"EVALUATION OF {model_name} WITH {data_set} DATA: COMPLETE\n")

  
  

def update_eval_df(evals_df, dataset_name, model_name, train_mse, train_r2, train_mae, test_mse, test_r2, test_mae, params_string):
    # Append a new row to the eval_df with params passed in
    evals_df = evals_df.append({
        'Dataset': dataset_name,
        'ML Model': model_name,
        'TRAIN MSE': train_mse,
        'TRAIN R2': train_r2,
        'TRAIN MAE': train_mae,
        'TEST MSE': test_mse,
        'TEST R2': test_r2,
        'TEST MAE': test_mae,
        'params_string': params_string
    }, ignore_index=True)
    # return updated df
    return evals_df

  
  
def update_pred_df(preds_df, dataset_name, model_name, Y_train_pred, Y_test_pred, params_string):
    preds_df = preds_df.append({
        'Dataset': dataset_name,
        'Model': model_name,
        'Train Prediction Results': Y_train_pred,
        'Test Prediction Results': Y_test_pred,
        'params_string': params_string
    }, ignore_index=True)
    # return updated df
    return preds_df
  
  
