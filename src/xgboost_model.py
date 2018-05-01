import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence


train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

y = train_data.SalePrice
X = train_data.drop(columns=['SalePrice'])
X_copy = X.copy()
# one hot encode
low_cardinality_cols = [col for col in X_copy.columns if
                        X_copy[col].nunique() < 10 and
                        X_copy[col].dtype == "object"]

numbers_cols = [col for col in X_copy.columns if
               X_copy[col].dtype in ['int64', 'float64']]


my_cols = numbers_cols + low_cardinality_cols
my_X = X_copy[my_cols]

one_hot_encode_X = pd.get_dummies(my_X)
one_hot_encode_X_copy = one_hot_encode_X

# add missing data
imputer = Imputer()
one_hot_encode_X_copy = pd.DataFrame(imputer.fit_transform(one_hot_encode_X_copy))
one_hot_encode_X_copy.columns = one_hot_encode_X.columns

train_X, test_X, train_y, test_y = train_test_split(one_hot_encode_X_copy.as_matrix(), y.as_matrix(), test_size=0.25)


# original model
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# Model Tuning
# my_model = XGBRegressor(n_estimators=1000)
# my_model.fit(train_X, train_y, early_stopping_rounds=5,
#              eval_set=[(test_X, test_y)], verbose=False)


# Model Tuning
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(train_X, train_y, early_stopping_rounds=5,
#              eval_set=[(test_X, test_y)], verbose=False)

my_plots = plot_partial_dependence(my_model,
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=train_X,            # raw predictors data.
                                   feature_names=['x`', 'Landsize', 'BuildingArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis

# make predictions
predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

def print_segment(seg_str):
    print("\n##### " + seg_str + " #####")

print_segment("train data key")
print(train_data.keys())

print_segment("x data keyss")
print(X.keys())

print_segment("x copy data keys")
print(X_copy.keys())

print_segment("one hot encode keys")
print(one_hot_encode_X.keys())

print_segment("one hot encode copy keys")
print(one_hot_encode_X_copy.keys())

