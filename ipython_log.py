# IPython log file

get_ipython().magic('logstart')
import encoding
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import encoding
import numpy as np
import joblib
from pathlib import Path
df = pd.read_csv("dbn_fem.txt", sep=" ")
df
df_test = pd.read_csv("dbn_fem.txt")
df_test
del(df_test)
df
df_test = pd.read_csv("dbn_fem.txt", header=None)
df
df = pd.read_csv("dbn_fem.txt", header=None)
df
df = pd.read_csv("dbn_fem.txt", sep=" ", header=None)
df
df is pd.DataFrame
3 is int
type(df)
df = df.rename(columns={"0":"lat", "1":"long", "2":"age", "3":"gender})
df = df.rename(columns={"0":"lat", "1":"long", "2":"age", "3":"gender"})
df
df.columns
df = df.rename(columns={0:"lat", 1:"long", 2:"age", 3:"gender"})
df
df.to_csv('cleaned_main_dataset.csv')
for count, element in enumerate(df["gender"]):
    if element == "Male":
        df["gender"][count] = 1
    else:
        df["gender"][count] = 0
    
df
df = pd.read_csv("cleaned_main_dataset.csv")
df
for count, element in enumerate(df["gender"]):
    if element == "male":
        df["gender"][count] = 1
    else:
        df["gender"][count] = 0
    
df
clean_df["sin_time"] = pd.Series(np.zeros(929))
clean_df["cos_time"] = pd.Series(np.zeros(929))
df["sin_time"] = pd.Series(np.zeros(929))
df["cos_time"] = pd.Series(np.zeros(929))
df
import math
np.sin(3 * 60)
np.cos(3 * 60)
np.random.normal(loc=3*60)
np.random.normal(loc=3*60)
np.random.normal(loc=3*60)
fake_times = np.random.normal(loc=180, size=928)
fake_times
df["lat"] = [encoding.cos_time(element) for element in fake_times]
len(fake_times)
len(df["lat"])
fake_times = np.random.normal(loc=180, size=929)
df["lat"] = [encoding.cos_time(element) for element in fake_times]
df
new_df = pd.read_csv("cleaned_main_dataset.csv")
df["lat"] = new_df["lat"]
df
df["sin_time"] = [encoding.cos_time(element) for element in fake_times]
df
df["sin_time"] = [encoding.sin_time(element) for element in fake_times]
df
df["cos_time"] = [encoding.cos_time(element) for element in fake_times]
df
fake_times
fake_times = np.random.normal(loc=180, size=929, scale=30)
fake_times
fake_times = np.random.normal(loc=180, size=929, scale=50)
fake_times = np.random.normal(loc=180, size=929, scale=100)
fake_times
fake_times = np.random.normal(loc=180, size=929, scale=50)
fake_times
df["sin_time"] = [encoding.sin_time(element) for element in fake_times]
df["cos_time"] = [encoding.cos_time(element) for element in fake_times]
df
del(df[0])
df
df.columns
del(df['Unnamed: 0'])
df
