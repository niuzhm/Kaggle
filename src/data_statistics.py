import pandas as pd
# save filepath to variable for easier access
melbourne_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
print(melbourne_data.describe())

print("\nprint data columns")
print(melbourne_data.columns)

print("\nprint LotArea")
print(melbourne_data.LotArea)


print("\n print head 5")
print(melbourne_data.head(5))
