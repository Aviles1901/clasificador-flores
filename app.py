import streamlit as st
import joblib
import numpy as np

model = joblib.load("mlp_model.joblib")
le = joblib.load("label_encoder.joblib")

st.title("Clasificador de Flores IRIS")

sepal_length = st.number_input("Largo del sepalo", min_value=0.0)
sepal_width = st.number_input("Ancho del sepalo", min_value=0.0)
petal_length = st.number_input("Largo del petalo", min_value=0.0)
petal_width = st.number_input("Ancho del petalo", min_value=0.0)

if st.button("Predecir"):
    datos = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediccion = model.predict(datos)
    especie = le.inverse_transform(prediccion)
    st.success(f"La especie predicha es: {especie[0]}")
