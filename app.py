# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st

import pandas as pd
import numpy as np
import prediction

# Presentation de l'application

st.title('Application : RevieWAnalyzer')

monlabel = "Quel texte analyser ? "
options = pd.DataFrame(['Avis dataset', 'Texte libre'])

n_topics = st.number_input(label="Le nombre de topics", min_value=1,max_value=15)  # st.number_input("topic number", value=0)#st.number_input(label= "Le nombre de topics", min_value=0, max_value=15)
#with st.sidebar:
st.radio(monlabel, options)
text = st.text_input(label="Donnez nous votre avis")

if st.button(label="Détecter le sujet d'insatisfaction") == True:
    prediction.predict_topics(prediction.model, prediction.vectoriseur, n_topics,text)
