# Refer:- https://medium.com/machine-learner/independent-and-dependent-variables-1-10d8553ad616
# Refer:- https://www.superdatascience.com/pages/machine-learning
# Refer:- https://www.udemy.com/course/data-science-for-beginners-hands-on-data-science-in-python

"""Data Preprocessing is a crucial step before making ML model. The model won't work properly without it. It can be a bit boring for some,
but is a necessary step in order to be able to work on a ML model."""
import numpy as np # Use for Sci. Operation and different operation on array
import pandas as pd # Use for data analysis, The pandas library is used to import and manage the datasets.
import matplotlib.pyplot as plt # Ploting lib in python, is use for data visualization

# Importing Dataset

dataset = pd.read_csv('data.csv') # Import file in dataframe in array with index in row and column

# Import independent variables

x = dataset.iloc[:, :-1].values

# Import dependent Variables

y = dataset.iloc[:, 3].values

# Working with missing data

from sklearn.impute import SimpleImputer # Use different ML models and data preprocessing, SimpleImputer use for data preprocessing that is work with missing data

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

