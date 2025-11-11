import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏭 Manufacturing Efficiency Prediction App")
st.write("Enter process parameters below to predict output using your trained Linear Regression model.")

# Feature input fields
Injection_Temperature = st.number_input("Injection Temperature")
Injection_Pressure = st.number_input("Injection Pressure")
Cycle_Time = st.number_input("Cycle Time")
Cooling_Time = st.number_input("Cooling Time")
Material_Viscosity = st.number_input("Material Viscosity")
Ambient_Temperature = st.number_input("Ambient Temperature")
Machine_Age = st.number_input("Machine Age")
Operator_Experience = st.number_input("Operator Experience")
Maintenance_Hours = st.number_input("Maintenance Hours")
Temperature_Pressure_Ratio = st.number_input("Temperature Pressure Ratio")
Total_Cycle_Time = st.number_input("Total Cycle Time")
Efficiency_Score = st.number_input("Efficiency Score")
Machine_Utilization = st.number_input("Machine Utilization")
Output_per_Cycle = st.number_input("Output per Cycle")
Cooling_to_Cycle_Ratio = st.number_input("Cooling to Cycle Ratio")
Temp_Pressure_Efficiency = st.number_input("Temp Pressure Efficiency")
Experience_per_MachineAge = st.number_input("Experience per Machine Age")
Maintenance_per_Part = st.number_input("Maintenance per Part")
Utilization_to_Efficiency = st.number_input("Utilization to Efficiency")

# Collect all inputs
input_data = np.array([[Injection_Temperature, Injection_Pressure, Cycle_Time, Cooling_Time,
                        Material_Viscosity, Ambient_Temperature, Machine_Age, Operator_Experience,
                        Maintenance_Hours, Temperature_Pressure_Ratio, Total_Cycle_Time, Efficiency_Score,
                        Machine_Utilization, Output_per_Cycle, Cooling_to_Cycle_Ratio, Temp_Pressure_Efficiency,
                        Experience_per_MachineAge, Maintenance_per_Part, Utilization_to_Efficiency]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"✅ Predicted Output: {prediction[0]:.2f}")
