import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Step 1: Load the data from an Excel file
df = pd.read_excel('FINAL_V3.xlsx', sheet_name='Sheet1')

# Step 2: Print the column names to inspect
print("Column Names:", df.columns)

# Step 3: Prepare the data for machine learning
# Assuming the columns for time and voltage are named 'Time_1', 'Voltage_1', 'Time_2', 'Voltage_2', ..., 'Time_1600', 'Voltage_1600'
time_columns = [f'Time_{i}' for i in range(1, 1301)]
voltage_columns = [f'Voltage_{i}' for i in range(1, 1301)]

# Combine time and voltage columns to make up 3200 features
X = df[time_columns + voltage_columns]  # 3200 features: Time and Voltage

# Target for classification (Material)
y_classification = df['Material']  # Target for classification (COIN or ERASER)

# Step 4: Convert height and distance to classification tasks (predict 10 or 30)
y_height = np.where(df['Height'] == 30, 1, 0)  # 1 for 30, 0 for 10
y_distance = np.where(df['Distance'] == 30, 1, 0)  # 1 for 30, 0 for 10

# Step 5: Split data into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_height_train, y_height_test, y_distance_train, y_distance_test = train_test_split(
    X, y_classification, y_height, y_distance, test_size=0.3, random_state=42
)

# Step 6: Train the Classification Model for Material Prediction
clf = RandomForestClassifier(n_estimators=500, random_state=60)
clf.fit(X_train, y_class_train)

# Step 7: Predict and evaluate the Classification Model for Material
y_class_pred = clf.predict(X_test)
classification_accuracy = accuracy_score(y_class_test, y_class_pred)
print(f"Classification Accuracy for Material: {classification_accuracy * 100:.2f}%")

# Step 8: Train the Classification Model for Height Prediction
height_clf = RandomForestClassifier(n_estimators=500, random_state=60)
height_clf.fit(X_train, y_height_train)

# Step 9: Predict and evaluate the Height Classification Model
y_height_pred = height_clf.predict(X_test)
height_accuracy = accuracy_score(y_height_test, y_height_pred)
print(f"Height Classification Accuracy: {height_accuracy * 100:.2f}%")

# Step 10: Train the Classification Model for Distance Prediction
distance_clf = RandomForestClassifier(n_estimators=500, random_state=60)
distance_clf.fit(X_train, y_distance_train)

# Step 11: Predict and evaluate the Distance Classification Model
y_distance_pred = distance_clf.predict(X_test)
distance_accuracy = accuracy_score(y_distance_test, y_distance_pred)
print(f"Distance Classification Accuracy: {distance_accuracy * 100:.2f}%")

# Step 12: Save the trained models to files
# Save the classification model for material
joblib.dump(clf, 'classification_model_FINALpls.pkl')

# Save the classification model for height
joblib.dump(height_clf, 'height_classification_model_FINALpls.pkl')

# Save the classification model for distance
joblib.dump(distance_clf, 'distance_classification_model_FINALpls.pkl')

print("Models saved as 'classification_model_FINALpls.pkl', 'height_classification_model_FINALpls.pkl', and 'distance_classification_model_FINALpls.pkl'.")


# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.model_selection import train_test_split
# import joblib
# import numpy as np

# # Step 1: Load the data from an Excel file
# df = pd.read_excel('TOTAL_DATASET_V1.xlsx', sheet_name='Sheet1')

# # Step 2: Print the column names to inspect
# print("Column Names:", df.columns)

# # Step 3: Dynamically generate column names for Time and Voltage
# time_columns = [col for col in df.columns if 'Time' in col]
# voltage_columns = [col for col in df.columns if 'Voltage' in col]

# # Step 4: Prepare the data for machine learning
# X = df[time_columns + voltage_columns]  # Time and Voltage values
# y_classification = df['Material']  # Target for classification (COIN or ERASER)
# y_regression = df[['Height', 'Distance']]  # Target for regression (Height and Distance)

# # Step 5: Ask the user for the train-test split percentage
# train_percentage = 99
# test_percentage = 100 - train_percentage  # Automatically calculate the test percentage

# # Step 6: Split the data into train and test sets
# X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
#     X, y_classification, y_regression, train_size=train_percentage/100, random_state=42
# )

# # Step 7: Print the split sizes for verification
# print(f"Training data size: {len(X_train)}")
# print(f"Test data size: {len(X_test)}")

# # Step 8: Train the Classification Model (Random Forest Classifier)
# clf = RandomForestClassifier(n_estimators=500, random_state=60)
# clf.fit(X_train, y_class_train)

# # Step 9: Predict and evaluate the Classification Model
# y_class_pred = clf.predict(X_test)
# classification_accuracy = accuracy_score(y_class_test, y_class_pred)
# print(f"\nClassification Accuracy: {classification_accuracy * 100:.2f}%")

# # Step 10: Train the Regression Model (Random Forest Regressor)
# reg = RandomForestRegressor(n_estimators=500, random_state=60)
# reg.fit(X_train, y_reg_train)

# # Step 11: Predict and evaluate the Regression Model
# y_reg_pred = reg.predict(X_test)
# regression_mse = mean_squared_error(y_reg_test, y_reg_pred)
# print(f"Regression MSE (Mean Squared Error): {regression_mse:.2f}")

# # Step 12: Save the trained models to files
# joblib.dump(clf, 'classification_model_FINAL1.pkl')
# joblib.dump(reg, 'regression_model_FINAL1.pkl')
# print("\nModels saved as 'classification_model_FINAL1.pkl' and 'regression_model_FINAL1.pkl'.")

# # Step 13: Initialize variables for the best thresholds
# best_height_threshold = None
# best_distance_threshold = None
# best_accuracy = 0  # We will track the best accuracy

# # Step 14: Test thresholds for height and distance from 10 to 30 with steps of 0.1
# step_size = 0.1
# for height_threshold in np.arange(10, 30, step_size):
#     for distance_threshold in np.arange(10, 30, step_size):
        
#         # Apply thresholding to predicted heights
#         height_thresholded = [30 if pred > height_threshold else 10 for pred in y_reg_pred[:, 0]]

#         # Apply thresholding to predicted distances
#         distance_thresholded = [30 if pred > distance_threshold else 10 for pred in y_reg_pred[:, 1]]

#         # Step 15: Calculate success rates for this pair of threshold values
#         # Material Prediction Success Rate
#         material_success_rate = sum(y_class_pred == y_class_test) / len(y_class_test)

#         # Height Prediction Success Rate (considering thresholded values)
#         height_success_rate = sum(np.array(height_thresholded) == y_reg_test['Height']) / len(y_reg_test)

#         # Distance Prediction Success Rate (considering thresholded values)
#         distance_success_rate = sum(np.array(distance_thresholded) == y_reg_test['Distance']) / len(y_reg_test)

#         # Step 16: Calculate the combined accuracy (simple average of height and distance accuracy)
#         accuracy = (height_success_rate + distance_success_rate) / 2

#         # Track the best accuracy and corresponding thresholds
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_height_threshold = height_threshold
#             best_distance_threshold = distance_threshold

# # Step 17: Output the best thresholds and accuracy
# print("\nBest Thresholds for Prediction:")
# print(f"Best Height Threshold: {best_height_threshold}")
# print(f"Best Distance Threshold: {best_distance_threshold}")
# print(f"Best Combined Accuracy: {best_accuracy * 100:.2f}%")

# # Step 18: Apply the best threshold values to the predictions
# height_thresholded = [30 if pred > best_height_threshold else 10 for pred in y_reg_pred[:, 0]]
# distance_thresholded = [30 if pred > best_distance_threshold else 10 for pred in y_reg_pred[:, 1]]

# # Step 19: Compare the predicted values with the actual values in the dataset
# actual_material = y_class_test
# actual_height = y_reg_test['Height']
# actual_distance = y_reg_test['Distance']

# # Step 20: Prepare a DataFrame for easy comparison
# predictions_df = pd.DataFrame({
#     'Index': X_test.index,
#     'Actual Material': actual_material,
#     'Predicted Material': y_class_pred,
#     'Actual Height': actual_height,
#     'Predicted Height': height_thresholded,
#     'Actual Distance': actual_distance,
#     'Predicted Distance': distance_thresholded
# })

# # Step 21: Calculate success rates for predictions
# material_success_rate = sum(y_class_pred == actual_material) / len(actual_material)
# height_success_rate = sum(height_thresholded == actual_height) / len(actual_height)
# distance_success_rate = sum(distance_thresholded == actual_distance) / len(actual_distance)

# print("\nSuccess Rates for Predictions:")
# print(f"Material Prediction Success Rate: {material_success_rate * 100:.2f}%")
# print(f"Height Prediction Success Rate: {height_success_rate * 100:.2f}%")
# print(f"Distance Prediction Success Rate: {distance_success_rate * 100:.2f}%")

# # Step 22: Write the results to an Excel file
# output_filename = 'predictions_results_with_best_thresholds.xlsx'
# with pd.ExcelWriter(output_filename) as writer:
#     predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
#     # Optionally, you can also write the success rates to a new sheet
#     success_rate_df = pd.DataFrame({
#         'Prediction Type': ['Material', 'Height', 'Distance'],
#         'Success Rate (%)': [material_success_rate * 100, height_success_rate * 100, distance_success_rate * 100]
#     })
#     success_rate_df.to_excel(writer, sheet_name='Success Rates', index=False)

# print(f"\nResults saved to {output_filename}")



#####______________________________________________________________________________________________________________________#####


# #improved
# import pandas as pd
# import joblib
# import numpy as np

# # Step 1: Load the data from an Excel file
# df = pd.read_excel('TOTAL_DATASET_V1.xlsx', sheet_name='Sheet1')

# # Step 2: Load the pre-trained models (Ensure you have saved the models)
# clf = joblib.load('classification_model_FINAL.pkl')  # Material prediction model
# reg = joblib.load('regression_model_FINAL.pkl')      # Height and Distance prediction model

# # Step 3: Define the row numbers you want to predict for
# indices_to_predict = [
#     608, 302, 621, 548, 375, 621, 548, 621, 548, 621,
#     621, 548, 375, 396, 328, 393, 78, 234
# ]

# # Step 4: Prepare the data for prediction (Only select relevant rows)
# rows_to_predict = df.iloc[indices_to_predict]

# # Step 5: Dynamically generate column names for Time and Voltage
# time_columns = [col for col in df.columns if 'Time' in col]
# voltage_columns = [col for col in df.columns if 'Voltage' in col]

# # Step 6: Prepare the feature data (Time and Voltage values)
# X_to_predict = rows_to_predict[time_columns + voltage_columns]

# # Step 7: Predict the material (COIN or ERASER)
# material_predictions = clf.predict(X_to_predict)

# # Step 8: Predict the height and distance
# height_distance_predictions = reg.predict(X_to_predict)

# # Step 9: Apply Thresholding to height and distance predictions
# height_threshold = 15
# distance_threshold = 15

# # Apply thresholding to predicted heights
# height_thresholded = [30 if pred > height_threshold else 10 for pred in height_distance_predictions[:, 0]]

# # Apply thresholding to predicted distances
# distance_thresholded = [30 if pred > distance_threshold else 10 for pred in height_distance_predictions[:, 1]]

# # Step 10: Compare the predicted values with the actual values in the dataset
# actual_material = rows_to_predict['Material']
# actual_height = rows_to_predict['Height']
# actual_distance = rows_to_predict['Distance']

# # Step 11: Prepare a DataFrame for easy comparison
# predictions_df = pd.DataFrame({
#     'Index': indices_to_predict,
#     'Actual Material': actual_material,
#     'Predicted Material': material_predictions,
#     'Actual Height': actual_height,
#     'Predicted Height': height_thresholded,
#     'Actual Distance': actual_distance,
#     'Predicted Distance': distance_thresholded
# })

# # Step 12: Print the comparison between actual and predicted values
# print(predictions_df)
