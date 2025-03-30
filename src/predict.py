import pickle
import pandas as pd

# Load trained model
with open("models/random_forest.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Sample input data (modify as needed)
input_data = pd.DataFrame([{
    "battery_power": 1200,
    "blue": 1,
    "clock_speed": 1.5,
    "dual_sim": 1,
    "fc": 5,
    "four_g": 1,
    "int_memory": 32,
    "m_dep": 0.8,
    "mobile_wt": 150,
    "n_cores": 4,
    "px_height": 1000,
    "px_width": 2000,
    "ram": 2048,
    "sc_h": 14,
    "sc_w": 8,
    "talk_time": 10,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}])

# Make prediction
prediction = model.predict(input_data)
print("Predicted Price Range:", prediction[0])
