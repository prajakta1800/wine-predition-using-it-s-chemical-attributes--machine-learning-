import streamlit as st
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
 
#initializing model stuff
wine_data = load_wine()



def logistic_regression(test_sample):
    model = LogisticRegression(solver='liblinear')
    model.fit(wine_data.data, wine_data.target)
    prediction = model.predict(test_sample.reshape(1,-1))
    return prediction

def k_nearest_neighbour(test_sample):
    model = KNeighborsClassifier(n_neighbors=50)
    model.fit(wine_data.data, wine_data.target)
    prediction = model.predict(test_sample.reshape(1,-1))
    return prediction

def random_forest(test_sample):
    model = RandomForestClassifier()
    model.fit(wine_data.data, wine_data.target)
    prediction = model.predict(test_sample.reshape(1,-1))
    return prediction

st.write("""
# Wine Recognition Model
## This Model Classifies a Wine based on it's Chemical Attributes
""")
st.markdown("""---""") 

options = ["Logistic Regression", "KNearest Neighbours", "Random Forest"]
choice = st.selectbox("Please Select a Classifier Algorithm:", options)

st.write(f"You selected **{choice}** Algorithm.")
st.markdown("""---""") 
st.write(f"Please Enter the **Chemical Attributes** of the Wine to Classify based on **{choice}** Algorithm")

alcohol = st.number_input("Alcohol")
malic_acid = st.number_input("Malic Acid")
ash = st.number_input("Ash")
alcalinity_of_ash = st.number_input("Alcalinity of Ash")
magnesium = st.number_input("Magnesium")
total_phenols = st.number_input("Total Phenols")
flavanoids = st.number_input("Flavanoids")
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols")
proanthocyanins = st.number_input("Proanthocyanins")
color_intensity = st.number_input("Color Intensity")
hue = st.number_input("Hue")
od280_od315_of_diluted_wines = st.number_input("OD280/OD315 of Diluted Wines")
proline = st.number_input("Proline")

input_values = [alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline]

st.write(f"Input Values: {input_values}")

result_class = [0]

if(choice == options[0]):
    result_class = logistic_regression(np.array(input_values).reshape(1,-1))
elif(choice == options[1]):
    result_class = k_nearest_neighbour(np.array(input_values).reshape(1,-1))
else:
    result_class = random_forest(np.array(input_values).reshape(1,-1))

st.markdown("""---""") 

st.write(f"### The Inputed Sample Belongs to Class {result_class[0]}.")
