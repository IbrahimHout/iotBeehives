import pandas as pd

df = pd.read_csv('analysis.csv')

# get the top  This will return the first few rows for the object based on columns values
print(df.head())

# get the data types of each column
print(df.info())

# get the summary statistics of the dataframe
print(df.describe())

# get the number of missing values in each column
print(df.isnull().sum())

# get the number of unique values in each column
print(df.nunique())
