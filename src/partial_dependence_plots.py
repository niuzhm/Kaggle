import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/melb_data.csv')
print(train_data.keys())
y = train_data.Price
X = train_data.drop(columns=['Id', 'Price'])
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
my_model = GradientBoostingRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y)

# Model Tuning
# my_model = XGBRegressor(n_estimators=1000)
# my_model.fit(train_X, train_y, early_stopping_rounds=5,
#              eval_set=[(test_X, test_y)], verbose=False)


# Model Tuning
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(train_X, train_y, early_stopping_rounds=5,
#              eval_set=[(test_X, test_y)], verbose=False)
names = ['Distance', 'Landsize', 'BuildingArea']
fig, axs = plot_partial_dependence(my_model,
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=train_X,            # raw predictors data.
                                   feature_names=names, # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis

fig.suptitle('Partial dependence of house value on nonlocation features\n'
             'for the Melbourne housing dataset')
plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

print('Custom 3d plot via ``partial_dependence``')
fig = plt.figure()

target_feature = (0, 2)
pdp, axes = partial_dependence(my_model, target_feature,
                               X=train_X, grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of house value on median\n'
             'age and average occupancy')
plt.subplots_adjust(top=0.9)

plt.show()