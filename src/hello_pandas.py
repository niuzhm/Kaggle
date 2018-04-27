import pandas as pd
melbourne_file_path = '../input/house-prices-advanced-regression-techniques/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']


print(melbourne_data.keys())
X = melbourne_data[melbourne_predictors]
print("X description")
print(X.describe())

# drop row with nan element
X = X.dropna()
print("after drop nan")
print(X.describe())

# get head 2 rows
print("print melbourne_data[:5]")
print(melbourne_data[:5])
print("transfer")
print(melbourne_data[:2].T)


# get element at i row and j column
print("print melbourne_data.iat[i,j]")
print(melbourne_data.iat[1, 0])

# get row between i row and j row (not include j)
print("print melbourne_data.iloc[0:1]")
print(melbourne_data.iloc[0:1])
print("print melbourne_data.iloc[0:2]")
print(melbourne_data.iloc[0:2])

# get row 0, 1 and column after 2
print("print melbourne_data.iloc[0:2, 2:]")
print(melbourne_data.iloc[0:2, 2:])

# get column of attribute as data frame
y = melbourne_data.Price
print("y description")
print(y.describe())

