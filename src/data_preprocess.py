import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Read the data
import pandas as pd
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
print("\n##### train data keys #####")
print(train_data.keys())

target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which drops these columns.
# For more detail (and a better approach) to missing values, see
# https://www.kaggle.com/dansbecker/handling-missing-values
cols_with_missing = [col for col in train_data.columns
                                 if train_data[col].isnull().any()]

# drop missing value columns
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
print("\n##### candidate_train_predictors keys #####")
print(candidate_train_predictors.keys())

candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)
print("\n##### candidate_test_predictors keys #####")
print(candidate_test_predictors.keys())


# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

# remove columns which unique value is bigger 10
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]
print("\n##### my_cols #####")
print(my_cols)
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

print("\n##### candidate test predictors head 5#####")
print(candidate_train_predictors.head(5))
print("\n##### one hot encoded training data head 5 #####")
print(one_hot_encoded_training_predictors.head(5))

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring='neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))



melbourne_data = pd.read_csv(melbourne_file_path)

# make copy to avoid changing original data (when Imputing)
new_data = melbourne_data.copy()
print("\n##### new data keys #####")
print(new_data.keys())

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns
                                 if new_data[col].isnull().any())

for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

print("\n##### new data keys after add missing #####")
print(new_data.keys())

# Imputation
my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data.select_dtypes(exclude=['object']))
print("\n##### new data keys after fit transforms #####")
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

# new_data = my_imputer.fit_transform(new_data)
# one hot encode