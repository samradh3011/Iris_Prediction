import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

st.set_page_config(page_title='Iris Prediction',
                   page_icon='https://lh3.googleusercontent.com/proxy/MLJ6TbXxlz7Bc9CQ7ND3AMWDs6gfQb72iCRX5IG-1ObhDWLOcdrKgVmezx2vBszGHMuFfYTwK7JE_XdWuZdZiC0LfVo8JA0')
st.write('''
# Predict ***IRIS*** Flower Species
''')
data = datasets.load_iris()
train_x = data.data
train_y = data.target
# ------------------------------------------------------
model = LogisticRegression(multi_class="multinomial", solver="newton-cg").fit(train_x, train_y)
input_data = pd.DataFrame()
s_length = st.sidebar.slider('Sepal Length (Centimeters)', min_value=1.00, max_value=8.00)
s_width = st.sidebar.slider('Sepal Width (Centimeters)', min_value=1.00, max_value=8.00)
p_length = st.sidebar.slider('Petal length (Centimeters)', min_value=1.00, max_value=8.00)
p_width = st.sidebar.slider('Petal Width (Centimeters)', min_value=1.00, max_value=8.00)
input_data['Sepal Length'] = [s_length]
input_data['Sepal Width'] = [s_width]
input_data['Petal Length'] = [p_length]
input_data['Petal Width'] = [p_width]
# ------------------------------------------------------
prediction = model.predict(input_data)
st.header('***Input Data***')
st.dataframe(input_data)
st.header('***Predicted Flower Species***')
if prediction == 0:
    prediction = 'Setosa'
elif prediction == 1:
    prediction = 'Versicolor'
elif prediction == 2:
    prediction = 'Virginica'
st.write(f'***{prediction}***')
