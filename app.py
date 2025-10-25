import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
import numpy as np

class MyNN(nn.Module):
    def __init__(self, input_features):
        super(MyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.model(x)

try:
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Error: `model_columns.joblib` not found. Please run the training script first.")
    st.stop()

try:
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Error: `scaler.joblib` not found. Please run the training script first.")
    st.stop()

input_size = len(model_columns)
model = MyNN(input_features=input_size)

try:
    model.load_state_dict(torch.load('csat_model_state.pth'))
    model.eval()
except FileNotFoundError:
    st.error("Error: `csat_model_state.pth` not found. Please run the training script first.")
    st.stop()
except RuntimeError as e:
    st.error(f"Model architecture mismatch. The saved model's structure does not match the 'MyNN' class in this script. Error: {e}")
    st.stop()


st.title('DeepCSAT: E-Commerce Satisfaction Prediction')
st.write("Enter the customer interaction details below to predict the CSAT score.")

response_time = st.slider('Response Time (Minutes)', 0, 500, 30)
handling_time = st.slider('Connected Handling Time (Minutes)', 0, 60, 15)
item_price = st.number_input('Item Price ($)', min_value=0.0, max_value=5000.0, value=50.0)
issue_hour = st.slider('Hour of Day Issue Reported (0-23)', 0, 23, 14)

channel = st.selectbox('Channel Name', ['channel_name_Inbound', 'channel_name_Outcall'])
category = st.selectbox('Issue Category', ['category_Order Related', 'category_Cancellation', 'category_Product Queries'])
product = st.selectbox('Product Category', ['Product_category_Electronics', 'Product_category_LifeStyle', 'Product_category_Home'])

if st.button('Predict CSAT Score'):
    
    input_data = pd.DataFrame(0.0, index=[0], columns=model_columns, dtype=np.float32)

    try:
        if channel in input_data.columns:
            input_data[channel] = 1.0
        if category in input_data.columns:
            input_data[category] = 1.0
        if product in input_data.columns:
            input_data[product] = 1.0
    except KeyError as e:
        st.warning(f"Note: One of the selected categories ({e}) was not in the original training data, proceeding with defaults.")

    input_data['Response_Time_minutes'] = response_time
    input_data['connected_handling_time'] = handling_time
    input_data['Item_price'] = item_price
    input_data['Issue_Hour'] = issue_hour
    
    if 'Issue_DayOfWeek' in input_data.columns:
        input_data['Issue_DayOfWeek'] = 3.0
    
    if 'Tenure Bucket' in input_data.columns:
        input_data['Tenure Bucket'] = 2.0
    
    numerical_cols_to_scale = [
        'Item_price', 'connected_handling_time', 'Response_Time_minutes', 
        'Issue_DayOfWeek', 'Issue_Hour', 'Tenure Bucket'
    ]
    numerical_cols_in_model = [col for col in numerical_cols_to_scale if col in model_columns]
    
    if numerical_cols_in_model:
        input_data[numerical_cols_in_model] = scaler.transform(input_data[numerical_cols_in_model])
    
    input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        
        confidence_tensor, predicted_class_tensor = torch.max(probabilities, dim=1)
        
        predicted_class = predicted_class_tensor.item()
        confidence = confidence_tensor.item()
        
        predicted_score = predicted_class + 1

    st.success(f'**Predicted CSAT Score: {predicted_score}**')
    st.write(f'**Model Confidence: {confidence*100:.2f}%**')
    
    if predicted_score > 4:
        st.balloons()
        
    st.subheader("Score Probabilities")
    prob_data = pd.DataFrame({
        'Score': [1, 2, 3, 4, 5],
        'Probability': probabilities.flatten().numpy()
    })
    st.bar_chart(prob_data.set_index('Score'))

