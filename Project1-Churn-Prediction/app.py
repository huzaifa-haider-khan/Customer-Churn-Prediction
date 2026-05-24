import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Customer Churn Prediction System')
st.markdown('Enter customer details below to predict churn probability.')

# Load model
@st.cache_resource
def load_model():
    with open('best_churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success('✅ Model loaded successfully!')

# Input form
st.header('Customer Information')
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Demographics')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

with col2:
    st.subheader('Services')
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])

with col3:
    st.subheader('Account')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 70.0)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'])

# Predict button
if st.button('🔍 Predict Churn', type='primary'):
    # Build input dataframe matching training features
    input_dict = {
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [tenure * monthly_charges],
        'gender_Male': [1 if gender == 'Male' else 0],
        'Partner_Yes': [1 if partner == 'Yes' else 0],
        'Dependents_Yes': [1 if dependents == 'Yes' else 0],
        'PhoneService_Yes': [1 if phone_service == 'Yes' else 0],
        'MultipleLines_No phone service': [1 if multiple_lines == 'No phone service' else 0],
        'MultipleLines_Yes': [1 if multiple_lines == 'Yes' else 0],
        'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
        'InternetService_No': [1 if internet_service == 'No' else 0],
        'OnlineSecurity_No internet service': [1 if online_security == 'No internet service' else 0],
        'OnlineSecurity_Yes': [1 if online_security == 'Yes' else 0],
        'OnlineBackup_No internet service': [0],
        'OnlineBackup_Yes': [0],
        'DeviceProtection_No internet service': [0],
        'DeviceProtection_Yes': [0],
        'TechSupport_No internet service': [0],
        'TechSupport_Yes': [0],
        'StreamingTV_No internet service': [0],
        'StreamingTV_Yes': [0],
        'StreamingMovies_No internet service': [0],
        'StreamingMovies_Yes': [0],
        'Contract_One year': [1 if contract == 'One year' else 0],
        'Contract_Two year': [1 if contract == 'Two year' else 0],
        'PaperlessBilling_Yes': [0],
        'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
        'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
        'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0],
    }

    input_df = pd.DataFrame(input_dict).astype(float)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    churn_prob = probability[1] * 100

    st.header('Prediction Result')
    col_a, col_b = st.columns(2)

    with col_a:
        if prediction == 1:
            st.error('🚨 HIGH RISK: Customer likely to churn!')
            st.metric('Churn Probability', f'{churn_prob:.1f}%')
        else:
            st.success('✅ LOW RISK: Customer likely to stay!')
            st.metric('Retention Probability', f'{100-churn_prob:.1f}%')

    with col_b:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=churn_prob,
            title={'text': 'Churn Risk %'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'red' if churn_prob > 50 else 'green'},
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 70], 'color': '#fff3cd'},
                    {'range': [70, 100], 'color': '#f8d7da'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.header('💡 Recommendations')
    if prediction == 1:
        st.warning('**Actions to retain this customer:**')
        if contract == 'Month-to-month':
            st.write('• 📄 Offer discount for switching to annual contract')
        if monthly_charges > 70:
            st.write('• 💰 Offer a loyalty discount on monthly charges')
        if tenure < 12:
            st.write('• 🎁 Provide new customer retention bonus')
        if internet_service == 'Fiber optic':
            st.write('• 🌐 Check service quality and offer upgrade incentives')
    else:
        st.info('Customer appears satisfied. Continue monitoring.')