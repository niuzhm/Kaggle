import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Read the data
import pandas as pd
melbourne_file_path = '../input/house-prices-advanced-regression-techniques/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in melbourne_data.columns if
                                melbourne_data[cname].nunique() < 10 and
                                melbourne_data[cname].dtype == "object"]

numeric_cols = [cname for cname in melbourne_data.columns if
                                melbourne_data[cname].dtype in ['int64', 'float64']]

# remove columns which unique value is bigger 10
my_cols = low_cardinality_cols + numeric_cols

train_predictors = melbourne_data[my_cols]

# one hot encode
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

print("\n##### melbourne_data keys#####")
print(melbourne_data.keys())
print("\n##### one_hot_encoded_training_predictors keys #####")
print(one_hot_encoded_training_predictors.keys())

# make copy to avoid changing original data (when Imputing)
one_hot_encoded_training_predictors_copy = one_hot_encoded_training_predictors.copy()

print("\n##### print one_hot_encoded_training_predictors_copy describe #####")
print(one_hot_encoded_training_predictors_copy.describe())

print("\n##### one_hot_encoded_training_predictors_copy head 5 after add missing #####")
print(one_hot_encoded_training_predictors_copy.head(5))

# Imputation, add default value for one_hot_encoded_training_predictors_copy
# default value is determinate by Imputer(strategy="")
my_imputer = Imputer()

# https://www.kaggle.com/dansbecker/handling-missing-values, in the Comments, Christopher Sardegna
# noticed that fit_transform return a type of ndarray, so use DataFrame() to transfer to DataFrame
one_hot_encoded_training_predictors_copy = \
    pd.DataFrame(my_imputer.fit_transform(one_hot_encoded_training_predictors_copy))

# one_hot_encoded_training_predictors_copy will lose the column titles.
# add columns titles
one_hot_encoded_training_predictors_copy.columns = one_hot_encoded_training_predictors.columns
print("\n##### new data head 5 after fit transforms #####")
print(one_hot_encoded_training_predictors_copy.head(5))



def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring='neg_mean_absolute_error').mean()
