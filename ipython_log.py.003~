# IPython log file

get_ipython().magic('logstart')
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import encoding
df = pd.read_csv("latest_dataset.csv")
clean_df = pd.DataFrame()
clean_df["id"] = df["id"]
clean_df["gender"] = df["gender"]
for count, element in enumerate(clean_pf["gender"]):
    if element == "Male":
        clean_pf["gender"][count] = 1
    else:
        clean_pf["gender"][count] = 0
for count, element in enumerate(clean_df["gender"]):
    if element == "Male":
        clean_df["gender"][count] = 1
    else:
        clean_df["gender"][count] = 0
clean_df
import numpy as np
clean_df["sin_time"] = pd.Series(np.zeros(1000))
for count, element in enumerate(df["military_time"]):
    clean_df["sin_time"][count] = encoding.sin_time(encoding.minutes_in_day(element))
    
for count, element in enumerate(df["military_time"]):
    clean_df["sin_time"][count] = encoding.sin_time(encoding.military_time_in_minutes_fn(element))
    
for count, element in enumerate(df["military_time"]):
    clean_df["sin_time"][count] = encoding.sin_time(encoding.military_time_in_minutes_fn(element))
    
for count, element in enumerate(df["military_time"]):
    clean_df["sin_time"][count] = encoding.sin_time(encoding.military_time_in_minutes_fn(element))
    
for count, element in enumerate(df["military_time"]):
    clean_df["sin_time"][count] = encoding.sin_time(encoding.military_time_in_minutes_fn(element))
    
for count, element in enumerate(df["military_time"]):
    clean_df["sin_time"][count] = encoding.sin_time(encoding.military_time_in_minutes_fn(element))
    
clean_df
for count, element in enumerate(df["military_time"]):
    clean_df["sin_time"][count] = encoding.sin_time(encoding.military_time_in_minutes_fn(element))
    
clean_df
clean_df["cos_time"] = pd.Series(np.zeros(1000))
for count, element in enumerate(df["military_time"]):
    clean_df["cos_time"][count] = encoding.cos_time(encoding.military_time_in_minutes_fn(element))
    
clean_df
df
df.columns
clean_df["age"] = df["age"]
df
clean_df
df.columns
clean_df["danger_intensity"] = df["danger_intensity"]
clean_df
features = pd.DataFrame(clean_df, columns=clean_df.columns[:-1])
features
labels = pd.DataFrame(clean_df, columns=clean_df.columns[-1])
labels = pd.DataFrame(clean_df, columns=[clean_df.columns[-1]])
labels
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)print('Average baseline error: ', round(np.mean(baseline_errors), 2))
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))
feature_list = list(features.columns)
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)# Calculate the absolute errors
errors = abs(predictions - test_labels)# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Use the forest's predict method on the test data
predictions = rf.predict(test_features) # Calculate the absolute errors
errors = abs(predictions - test_labels) # Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
predictions
test_labels
abs(predictions - test_labels)
abs(list(predictions - test_labels))
predictions - test_labels
predictions
predictions - list(test_labels)
predictions - np.asarray(test_labels)
abs(predictions - np.asarray(test_labels))
# Use the forest's predict method on the test data
predictions = rf.predict(test_features) # Calculate the absolute errors
errors = abs(predictions - np.asarray(test_labels)) # Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / np.asarray(test_labels)) # Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
mape
errors
test_labels
np.asarray(test_labels)
np.squeeze()
np.squeeze(a=0)
np.squeeze(0)
np.squeeze(test_labels)
test_labels
np.asarray(np.squeeze(test_labels))
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / np.asarray(np.squeeze(test_labels))) # Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
mape
accuracy = 100 - np.mean(mape)
accuracy
mape.shape
errors.shape
errors
np.mean(mape)
test_labels
predictions.shape
predictions
errors = abs(predictions - np.asarray(np.sqeeze(test_labels)))
errors = abs(predictions - np.asarray(np.squeeze(test_labels)))
errors.shape
errors
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / np.asarray(np.squeeze(test_labels)))# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
mape
mape.shape
np.mean(mape)
np.asarray(np.squeeze(test_labels))
arrray_test_labels = _
[print(index, arrray_test_labels[index]) for x in arrray_test_labels if x > 1]
errors.shape
arrray_test_labels.shape
errors
[print(index, errors[index]) for x in errors if x > 1]
# Calculate mean absolute percentage error (MAPE)
ape = 100 * (errors / arrray_test_labels) # Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
errors / arrray_test_labels
errors ./ arrray_test_labels
[x/y for x, y in errors, arrray_test_labels]
[errors[x] / arrray_test_labels[x] for x in range(len(errors))]
errors[-6]
arrray_test_labels[-6]
predictions[-6]
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10000, random_state = 42)# Train the model on training data
rf.fit(train_features, train_labels);
