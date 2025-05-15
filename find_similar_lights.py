import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# loading the data
file_path = "Chatbot_Data_Final_SampleV1.csv"  #thr csv file
df = pd.read_csv(file_path)

#Preprocessing the data
required_columns = ["Reported_Minimum_Input_Voltage", "Lumens", "Input_Wattage", "CRI", "Manufacturer"]
df_filtered = df.dropna(subset=required_columns).copy()

#Encode Manufacturer
le = LabelEncoder()
df_filtered["Manufacturer_Encoded"] = le.fit_transform(df_filtered["Manufacturer"])

#Prepairing Feature Matrix 
features = ["Reported_Minimum_Input_Voltage", "Lumens", "Input_Wattage", "CRI", "Manufacturer_Encoded"]
X = df_filtered[features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Fittng knn Model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled)

# input new light spces CHANGE THIS COLUMN:
input_specs = {
    "Reported_Minimum_Input_Voltage": 100,
    "Lumens": 35000,
    "Input_Wattage": 400,
    "CRI": 80,
    "Manufacturer": "syska"
}

#Encode manufacturer
if input_specs["Manufacturer"] in le.classes_:
    manufacturer_encoded = le.transform([input_specs["Manufacturer"]])[0]
else:
    manufacturer_encoded = -1  # Unknown manufacturer

input_vector = np.array([[input_specs["Reported_Minimum_Input_Voltage"],
                          input_specs["Lumens"],
                          input_specs["Input_Wattage"],
                          input_specs["CRI"],
                          manufacturer_encoded]])
input_vector_scaled = scaler.transform(input_vector)

#Finding Nearest Matches 
distances, indices = knn.kneighbors(input_vector_scaled)
matches = df_filtered.iloc[indices[0]]

#displaying Results
print("\nTop Similar Light Specs:")
print(matches[["Manufacturer", "Model_Number", "Reported_Minimum_Input_Voltage",
               "Lumens", "Input_Wattage", "CRI"]].to_string(index=False))


## to find the products of your match , open terminal and run:
## python find_similar_lights.py
 