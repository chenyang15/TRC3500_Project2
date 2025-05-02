import serial
import time
import numpy as np
import joblib
import csv
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Macros
DETECT_MAX_DISTANCE = False  # Set to True if you want to detect maximum distance

# Setup serial communication
try:
    ser = serial.Serial('COM11', 912600, timeout=1)
    print(f"Serial port opened: {ser.name}")
except Exception as e:
    print(f"Failed to open serial port: {e}")
    exit()

time.sleep(2)  # Allow time for the serial device to reset

# Load the trained models (Ensure models are saved as 'classification_model.pkl' and 'regression_model.pkl')
clf = joblib.load('classification_model_FINALpls.pkl')  # Material prediction model
height = joblib.load('height_classification_model_FINALpls.pkl')      # Height and Distance prediction model
distance = joblib.load('distance_classification_model_FINALpls.pkl')      # Height and Distance prediction model
# clf = joblib.load('classification_model_SURFACE.pkl')  # Material prediction model
# height = joblib.load('height_classification_model_SURFACE.pkl')      # Height and Distance prediction model
# distance = joblib.load('distance_classification_model_SURFACE.pkl')      # Height and Distance prediction model

# Constants
if DETECT_MAX_DISTANCE:
    TRIGGER_VOLTAGE = 0.4  # TODO TBD test the maximum distance voltage to trigger without being too small
else:
    TRIGGER_VOLTAGE = 0.75  # 0.5volts

MAX_COLLECTION_TIME = 0.6  # TODO change when testing on lab table (0.6seconds)
NUM_SAMPLES = 1300  # We will collect 1600 samples

# Threshold values for height and distance predictions
HEIGHT_THRESHOLD = 23.79999999999995  # Height threshold for thresholding
DISTANCE_THRESHOLD = 15.69999999999998  # Distance threshold for thresholding

# Open CSV file for writing results
csv_filename = "final_measurements.csv"

# Write the header to the CSV file (add columns for time and voltage)
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)

# Function to collect data and make predictions
def collect_and_predict():
    voltageArray = []
    timeArray = []

    print("Waiting for trigger voltage...")

    # Wait for the trigger voltage to start collecting data
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            try:
                if line.startswith("Voltage Value:"):
                    voltage = float(line.split(":")[1].strip())
                    if voltage > TRIGGER_VOLTAGE:
                        if DETECT_MAX_DISTANCE:
                            print("Trigger voltage detected! Drop Detected!")
                        else:
                            print("Trigger voltage detected! Starting collection...")
                        startTime = time.time()
                        voltageArray.append(voltage)
                        timeArray.append(0.0)  # Start at time zero
                        break
            except ValueError:
                print(f"Invalid data: {line}. Retrying...")

    if not(DETECT_MAX_DISTANCE):
        print(f"Collecting data (max {MAX_COLLECTION_TIME} seconds)...")
        while (time.time() - startTime) < MAX_COLLECTION_TIME:
            if len(voltageArray) < NUM_SAMPLES:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    try:
                        if line.startswith("Voltage Value:"):
                            voltage = float(line.split(":")[1].strip())
                            elapsed = time.time() - startTime
                            voltageArray.append(voltage)
                            timeArray.append(elapsed)
                    except ValueError:
                        print(f"Invalid data: {line}. Retrying...")

        print(f"Collection complete. Got {len(voltageArray)} samples.")
        
        # Ensure we have the expected 1600 samples
        if len(voltageArray) < NUM_SAMPLES:
            print(f"Not enough data collected! Only {len(voltageArray)} samples.")
            return

        # Prepare the data to feed into the model
        # Concatenate the time and voltage arrays into a single 1D array of 3200 features (time + voltage)
        X_input = np.concatenate([np.array(timeArray[:NUM_SAMPLES]), np.array(voltageArray[:NUM_SAMPLES])])

        # Reshape to make it compatible with the model
        X_input = X_input.reshape(1, -1)  # Reshaping to 2D array with 1 row and 3200 columns

        # Predict the material (classification)
        predicted_material = clf.predict(X_input)

        # Predict the height and distance (regression)
        predicted_height = height.predict(X_input)
        predicted_distance = distance.predict(X_input)

        if (predicted_height == 0):
            predicted_height = 10
        else:
            predicted_height = 30

        if (predicted_distance == 0):
            predicted_distance = 10
        else:
            predicted_distance = 30

        # # Unpack the predicted values
        # predicted_height = predicted_values[:, 0]  # First column is predicted height
        # predicted_distance = predicted_values[:, 1]  # Second column is predicted distance

        # # Apply thresholding for height prediction
        # if predicted_height > HEIGHT_THRESHOLD:
        #     thresholded_height = 30  # Height above threshold
        # else:
        #     thresholded_height = 10  # Height below threshold

        # # Apply thresholding for distance prediction
        # if predicted_distance > DISTANCE_THRESHOLD:
        #     thresholded_distance = 30  # Distance above threshold
        # else:
        #     thresholded_distance = 10  # Distance below threshold

        # Output the predictions
        print(f"Predicted Material: {predicted_material[0]}")
        print(f"Predicted Height: {predicted_height}")
        print(f"Predicted Distance: {predicted_distance}")

        # Write the results to the CSV file with voltage and time arrays
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the result with voltage and time values as separate cells
            writer.writerow(timeArray[0:1300] + voltageArray[0:1300])  # Write each result

        # Plot the voltageArray against the timeArray with markers for the first and second maxima
        plt.figure(figsize=(10, 6))
        plt.plot(timeArray, voltageArray, label='Voltage vs Time', color='blue')  # Plot the voltage over time
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage vs Time Plot')

        # Add annotation
        annotation_text = f"Material: {predicted_material}\nPrediction: H{predicted_height}xD{predicted_distance}"
        plt.annotate(annotation_text, 
             xy=(0.05, 0.95), 
             xycoords='axes fraction', 
             fontsize=12, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.grid(True)
        plt.legend()
        plt.show()
        # time.sleep(2)
        # plt.close()
# Infinite loop to continuously collect data and predict for each new object drop
try:
    while True:
        collect_and_predict()  # Collect data and predict for each drop
        print("\nReady for the next object drop...\n")  # Prepare for the next drop
        time.sleep(1)  # Optional delay to avoid overwhelming the serial port
except KeyboardInterrupt:
    print("User interrupted")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    ser.close()
    print("Serial port closed")







# #internal classifier
# import pandas as pd
# import joblib
# import numpy as np

# # Step 1: Load the data from an Excel file
# df = pd.read_excel('MAGE_ADARE_TOTAL_V1.xlsx', sheet_name='Sheet1')

# # Step 2: Load the pre-trained models (Ensure you have saved the models)
# clf = joblib.load('classification_model_mage.pkl')  # Material prediction model
# reg = joblib.load('regression_model_mage.pkl')      # Height and Distance prediction model

# # Step 3: Prepare the data for prediction (Use all rows in the dataset)
# # Dynamically generate column names for Time and Voltage
# time_columns = [col for col in df.columns if 'Time' in col]
# voltage_columns = [col for col in df.columns if 'Voltage' in col]

# # Step 4: Prepare the feature data (Time and Voltage values for all rows)
# X_to_predict = df[time_columns + voltage_columns]

# # Step 5: Predict the material (COIN or ERASER)
# material_predictions = clf.predict(X_to_predict)
# # print(material_predictions)

# # Step 6: Predict the height and distance
# height_distance_predictions = reg.predict(X_to_predict)

# # Step 7: Initialize variables for the best thresholds
# best_height_threshold_eraser = None
# best_distance_threshold_eraser = None
# best_height_threshold_coin = None
# best_distance_threshold_coin = None
# best_accuracy_eraser = 0  # We will track the best accuracy for eraser
# best_accuracy_coin = 0  # We will track the best accuracy for coin

# # Step 8: Test thresholds for height and distance from 10 to 30 with steps of 0.1
# step_size = 0.3
# for height_threshold in np.arange(10, 30, step_size):
#     for distance_threshold in np.arange(10, 30, step_size):
#         count_eraser = 0
#         count_coin = 0
#         height_success_rate_eraser = 0
#         distance_success_rate_eraser = 0
#         height_success_rate_coin = 0
#         distance_success_rate_coin = 0
#         # Apply thresholding to predicted heights
#         height_thresholded = [30 if pred > height_threshold else 10 for pred in height_distance_predictions[:, 0]]

#         # Apply thresholding to predicted distances
#         distance_thresholded = [30 if pred > distance_threshold else 10 for pred in height_distance_predictions[:, 1]]

#         # Step 9: Calculate success rates for this pair of threshold values
#         # Material Prediction Success Rate
#         material_success_rate = sum(material_predictions == df['Material']) / len(df)
#         # print("before material for")
#         for i in range(len(material_predictions)):
#             if (material_predictions[i] == 'ERASER'):
#                 count_eraser += 1
#                 # Height Prediction Success Rate (considering thresholded values)
#                 height_success_rate_eraser += (np.array(height_thresholded)[i] == df['Height'][i])

#                 # Distance Prediction Success Rate (considering thresholded values)
#                 distance_success_rate_eraser += (np.array(distance_thresholded)[i] == df['Distance'][i])

#                 height_success_rate_eraser /= count_eraser
#                 distance_success_rate_eraser /= count_eraser
#             else:
#                 count_coin += 1
#                 # Height Prediction Success Rate (considering thresholded values)
#                 height_success_rate_coin += (np.array(height_thresholded)[i] == df['Height'][i])

#                 # Distance Prediction Success Rate (considering thresholded values)
#                 distance_success_rate_coin += (np.array(distance_thresholded)[i] == df['Distance'][i])

            
#                 height_success_rate_coin /= count_coin
#                 distance_success_rate_coin /= count_coin

#             # Step 10: Calculate the combined accuracy (simple average of height and distance accuracy)
#             accuracy_eraser = (height_success_rate_eraser + distance_success_rate_eraser) / 2
#             accuracy_coin = (height_success_rate_coin + distance_success_rate_coin) / 2

#             # print()
#             # Track the best accuracy and corresponding thresholds
#             if accuracy_eraser > best_accuracy_eraser:
#                 best_accuracy_eraser = accuracy_eraser
#                 best_height_threshold_eraser = height_threshold
#                 best_distance_threshold_eraser= distance_threshold
            
#             if accuracy_coin > best_accuracy_coin:
#                 best_accuracy_coin = accuracy_coin
#                 best_height_threshold_coin = height_threshold
#                 best_distance_threshold_coin= distance_threshold

# # Step 11: Output the best thresholds and accuracy
# print("\nBest Thresholds for Prediction:")
# print(f"Best Height Threshold for Coin: {best_height_threshold_coin} for Eraser: {best_height_threshold_eraser}")
# print(f"Best Distance Threshold for Coin: {best_distance_threshold_coin} for Eraser: {best_distance_threshold_eraser}")
# print(f"Best Combined Accuracy for Coin: {best_accuracy_coin * 100:.2f}% for Eraser: {best_accuracy_eraser * 100:.2f}%")

# for i in range(len(material_predictions)):    
#     # Step 12: Apply the best threshold values to the predictions
#     if (material_predictions[i]=="ERASER"):
#         pred = height_distance_predictions[i,0]
#         height_thresholded[i] = [30 if pred > best_height_threshold_eraser else 10 ]
#         pred = height_distance_predictions[i,1]
#         distance_thresholded[i] = [30 if pred > best_distance_threshold_eraser else 10]
#         # height_thresholded[i] = [30 if pred > best_height_threshold_eraser else 10 for pred in height_distance_predictions[:, 0]]
#         # distance_thresholded[i] = [30 if pred > best_distance_threshold_eraser else 10 for pred in height_distance_predictions[:, 1]]
#     else:
#         pred = height_distance_predictions[i,0]
#         height_thresholded[i] = [30 if pred > best_height_threshold_coin else 10 ]
#         pred = height_distance_predictions[i,1]
#         distance_thresholded[i] = [30 if pred > best_distance_threshold_coin else 10]


# # Step 13: Compare the predicted values with the actual values in the dataset
# actual_material = df['Material']
# actual_height = df['Height']
# actual_distance = df['Distance']

# # Step 14: Prepare a DataFrame for easy comparison
# predictions_df = pd.DataFrame({
#     'Index': df.index,
#     'Actual Material': actual_material,
#     'Predicted Material': material_predictions,
#     'Actual Height': actual_height,
#     'Predicted Height': height_thresholded,
#     'Actual Distance': actual_distance,
#     'Predicted Distance': distance_thresholded
# })
# material_predictions_array = np.array(material_predictions)
# height_thresholded_array = np.array(height_thresholded)
# distance_thresholded_array = np.array(distance_thresholded)

# # actual_material_array = np.array([actual_material.to_list])
# # actual_material_array=actual_material_array.reshape(-1,1)
# actual_height_array = np.array([actual_height.to_list()])
# actual_height_array=actual_height_array.reshape(-1,1)
# actual_distance_array = np.array([actual_distance.to_list()])
# actual_distance_array=actual_distance_array.reshape(-1,1)
# # print(f"material pred : {np.shape(hi)}  actual mat: {np.shape(niggo)}")
# # Step 15: Calculate success rates for predictions
# material_success_rate = np.sum(material_predictions_array == actual_material) / len(actual_material)
# height_success_rate = np.sum(height_thresholded_array == actual_height_array) / len(actual_height_array)
# distance_success_rate = np.sum(distance_thresholded_array == actual_distance_array) / len(actual_distance_array)


# # print(f"3. {(actual_material)}")


# print("\nSuccess Rates for Predictions:")
# print(f"Material Prediction Success Rate: {material_success_rate * 100:.2f}%")
# print(f"Height Prediction Success Rate: {height_success_rate * 100:.2f}%")
# print(f"Distance Prediction Success Rate: {distance_success_rate * 100:.2f}%")

# # Step 16: Write the results to an Excel file
# output_filename = 'predictions_results_with_thresholds_3.xlsx'
# with pd.ExcelWriter(output_filename) as writer:
#     predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
#     # Optionally, you can also write the success rates to a new sheet
#     success_rate_df = pd.DataFrame({
#         'Prediction Type': ['Material', 'Height', 'Distance'],
#         'Success Rate (%)': [material_success_rate * 100, height_success_rate * 100, distance_success_rate * 100]
#     })
#     success_rate_df.to_excel(writer, sheet_name='Success Rates', index=False)

# print(f"\nResults saved to {output_filename}")
