import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title="Salary Prediction",layout="centered")

model=joblib.load("Salary_training_model.pkl")
st.title("Salary Prediction App")

st.write("predict the salary based on years of professional experience")
st.sidebar.header("employee information")
exp=st.sidebar.number_input("years of experience",min_value=20,max_value=60,step=1)
if st.button("predict salary"):
    input=np.array([[exp]])
    pred=model.predict(input)
    #to display the result
    st.success(f"predicted salary = {pred}")