# Model to predict emergency occurrence to user
# Possible features used: current_military_time, day_of_week, date, location, age, gender
# More features could be added to the model as the system collects
# more data, increasing the possible accuracy and bias of the model
# Output: "Probability" of emergency occuring to the user

# Convert location data into something more useful?
# Replace location with a feature that calculates cumulative proximity?
# Location danger rating: a custom algorithm that simply adds a danger
# point for every emergency based location in dataset, weighted more towards the closer locations.
# final features: location_danger, cyclically encoded current_military_time (as sin and cos), one hot encoded date, encoded age, binarized gender


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import encoding
import numpy as np
import joblib

filename = "model.save"


# Clean dataset
def clean_dataset() -> pd.Dataframe:
    df = pd.read_csv("latest_dataset.csv")
    clean_df = pd.DataFrame()
    clean_df["id"] = df["id"]
    clean_df["gender"] = df["gender"]

    for count, element in enumerate(clean_df["gender"]):
        if element == "Male":
            clean_df["gender"][count] = 1
        else:
            clean_df["gender"][count] = 0

    clean_df["sin_time"] = pd.Series(np.zeros(1000))
    clean_df["cos_time"] = pd.Series(np.zeros(1000))

    for count, element in enumerate(df["military_time"]):
        clean_df["cos_time"][count] = encoding.cos_time(encoding.military_time_in_minutes_fn(element))
    for count, element in enumerate(df["military_time"]):
        clean_df["sin_time"][count] = encoding.sin_time(encoding.military_time_in_minutes_fn(element))

    clean_df["age"] = df["age"]
    clean_df["danger_intensity"] = df["danger_intensity"]
    return clean_df

# Train model on existing mock dataset
def train(clean_df):
    features = pd.DataFrame(clean_df, columns=clean_df.columns[:-1])
    labels = pd.DataFrame(clean_df, columns=[clean_df.columns[-1]])
    feature_list = list(features.columns)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
    rf.fit(train_features, train_labels);

    # Save model
    joblib.dump(rf, filename)


# Predict
def predict(encoded_features):
    model = joblib.load()
    return model.predict(encoded_features)
