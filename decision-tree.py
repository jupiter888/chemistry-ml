from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
# X_train, Y_train, X_test, Y_test = data_load("/Users/njord888/Desktop/ml_project/DATA/toxi/ipc81-train_data.XLSX","/Users/njord888/Desktop/ml_project/DATA/toxi/ipc81-test_data.XLSX")
from novy_load import data_loader
X_train_h, Y_train_h, X_test_h, Y_test_h = data_loader("/DATA/hcap/cpjk1-train_data.xlsx", "/DATA/hcap/cpjk1-test_data.xlsx")

# Create model
model_h = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
# Fit model
model_h.fit(X_train_h, Y_train_h)

# Predict the test set labels(Make predictions on the X_test set
Y_pred_test_h = model_h.predict(X_test_h)
# Predict the train set labels(Make predictions on the X_train set)
Y_pred_train_h = model_h.predict(X_train_h)

from eval_fx import eval_results
# RETURNS data_set, model_name, train_mse, train_r2, train_mae, test_mse, test_r2, test_mae
eval_results(model_h, "hcap", "Decision Tree", Y_train_h, Y_test_h, Y_pred_train_h, Y_pred_test_h)
