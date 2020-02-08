# IPython log file

get_ipython().magic('logstart')
import pandas as pd
import numpy as np
pd.read_csv("dataset_v2_with_encoded_normalized_fake_time_and_lat_long.csv")
df = pd.read_csv("dataset_v2_with_encoded_normalized_fake_time_and_lat_long.csv")
del(df["lat"])
del(df["long"])
df
df.to_csv("dataset_v3.csv", index=None)
np.random.uniform(size=100)
np.random.uniform(size=929)
np.random.uniform(size=100)
np.random.uniform(size=929)
temp np.random.uniform(size=929)
temp = np.random.uniform(size=929)
df["intensity"] = [x for x in temp]
df
df.to_csv("dataset_v4_with_intensity.csv", index=None)
def train(clean_df):
    features = pd.DataFrame(clean_df, columns=clean_df.columns[:-1])
    labels = pd.DataFrame(clean_df, columns=[clean_df.columns[-1]])
    feature_list = list(features.columns)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
    rf.fit(train_features, train_labels);

    # Save model, overwriting any previous model
    joblib.dump(rf, filename)
filename = "model.save"
train(df)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
train(df)
def predict(encoded_features):
    model = joblib.load()
    return model.predict(encoded_features)
np.asarray(df)[0]
predict(_)
predict(np.asarray(df)[0])
# Predict
def predict(encoded_features):
    model = joblib.load(filename)
    return model.predict(encoded_features)
predict(np.asarray(df)[0])
predict(np.asarray(df)[0].reshape(1,-1))
np.asarray(df)[0].reshape(1,-1)
np.asarray(df)[0].reshape(1,-1)[:-1]
np.asarray(df)[0][:-1].reshape(1,-1)
predict(_)
df[0]
np.asarray(df)[0]
float())
float(_)
np.asarray(df)[0]
np.asarray(df)[0][0]
np.asarray(df)[0][:-1].reshape(1,-1)
predict(_)
predict(np.asarray(df)[0][:-1].reshape(1,-1))
float(_)
# Predict
def predict(encoded_features: list) -> float:
    saved_model = joblib.load(filename)
    return float(saved_model.predict(encoded_features))
predict(np.asarray(df)[0][:-1].reshape(1,-1))
