import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
housing = pd.read_csv("Book1.csv")
print(housing.head())
print(housing.info())
print(housing['CHAS'].value_counts())

Manual process for splitting the data:------------------
def split_train_test(data,test_ratio):
    np.random.seed(2)
    shuffle = np.random.permutation(len(data))
    print(shuffle)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffle[:test_set_size]
    train_indices = shuffle[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set,test_set = split_train_test(housing,0.2)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")
---------------------------------------------------------------------------------------
# Splitting the data can be done by sklearn with an easy way
train_set,test_set = train_test_split(housing, test_size=0.2, random_state=2)
print(train_set)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")


#For equal splitting of data for specific column
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=2)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_train_set['CHAS'].value_counts())
print(strat_train_set)


from pandas.plotting import scatter_matrix
attributes = ['MEDV','RM', 'ZN', 'LSTAT']
print(scatter_matrix(housing[attributes], figsize=(12, 10)))
plt.show()

housing['TAXRM'] = housing['TAX']/housing['RM']
housing['INDUSZN'] = housing['INDUS']/housing['ZN']
print(housing.head())
corr_matrix = housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending= False))

# To take care of missing data in the dataset we have 3 options:-
# Get rid of the missing data points
# Get rid of the whole attribute
# Replace the missing values with (0, mean or median)
#------------------------------------------------
# option 1:
temp = housing.dropna(subset=['RM'])
print(temp.shape)
# -----------------------------------
# option 2:
housing.drop("RM", axis=1).shape
#---------------------------------------
# option 3:
median = housing['RM'].median()
print(median)
housing['RM'].fillna(median)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
print(imputer.statistics_)


# In all the options the original housing data is not changed. Copy is made
#-------------------------------------------
# Above process can be done by SimpleImputer of Sklearn library.
#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy= "median")
print(imputer.fit(housing))


my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])
housing_tr = my_pipeline.fit_transform(housing)
print(housing_tr)