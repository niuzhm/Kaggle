from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
melbourne_file_path = '../input/house-prices-advanced-regression-techniques/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

print("\n##### X description #####")
print(melbourne_data.describe())

melbourne_data = melbourne_data.dropna()

print("\n##### after drop nan #####")
print(melbourne_data.describe())

melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_predictors]

# get column of attribute as data frame
y = melbourne_data.Price

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


def get_mae_for_decision_tree(max_leaves_count, train_set, train_value, prediction_set, prediction_value):
    print("build decision tree")
    melbourne_model = DecisionTreeRegressor(max_leaf_nodes=max_leaves_count, random_state=0)
    melbourne_model.fit(train_set, train_value)
    val_predictions = melbourne_model.predict(prediction_set)
    val_error = mean_absolute_error(val_predictions, prediction_value)
    return val_error


def get_mae_for_randow_forest_tree(train_set, train_value, prediction_set, prediction_value):
    print("build random forest tree")
    melbourne_model = RandomForestRegressor()
    melbourne_model.fit(train_set, train_value)
    val_predictions = melbourne_model.predict(prediction_set)
    val_error = mean_absolute_error(val_predictions, prediction_value)
    return val_error


test_leaves_counts = [5, 50, 500, 5000]
# test_leaves_counts = range(100, 1000, 100)
for leaves_count in test_leaves_counts:
    model_error = get_mae_for_decision_tree(leaves_count, train_X, train_y, val_X, val_y)
    print("\n#####  leaves count " + str(leaves_count) + " prediction error is: #####")
    print(model_error)

print("mae of decision tree is:")
print(get_mae_for_decision_tree(500, train_X, train_y, val_X, val_y))

print("mae of decision tree is:")
print(get_mae_for_randow_forest_tree(train_X, train_y, val_X, val_y))
