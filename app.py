import streamlit as st
import pandas as pd
import joblib

# ==========================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================

st.set_page_config(
    page_title="Predicción de Churn",
    page_icon="📊",
    layout="centered"
)

# ==========================
# CARGA DEL MODELO
# ==========================

modelo = joblib.load("modelo_churn.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# TÍTULO
# ==========================

st.title("📊 Predicción de Abandono de Clientes")

st.markdown("""
Ingrese los datos del cliente para estimar la probabilidad
de abandono del servicio de telecomunicaciones.
""")

# ==========================
# ENTRADAS
# ==========================

tenure = st.slider(
    "Meses de permanencia (tenure)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Cargo mensual (Monthly Charges)",
    min_value=0.0,
    max_value=500.0,
    value=70.0,
    step=0.1
)

total_charges = st.number_input(
    "Cargo total acumulado (Total Charges)",
    min_value=0.0,
    value=1000.0,
    step=0.1
)

# ==========================
# BOTÓN DE PREDICCIÓN
# ==========================

if st.button("Predecir"):

    datos = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Escalar datos
    datos_escalados = scaler.transform(datos)

    # Predicción
    prediccion = modelo.predict(datos_escalados)

    # Probabilidad
    probabilidades = modelo.predict_proba(datos_escalados)

    prob_churn = probabilidades[0][1]

    st.subheader("Resultado")

    if prediccion[0] == 1:

        st.error(
            f"⚠️ Cliente con riesgo de abandono."
        )

    else:

        st.success(
            f"✅ Cliente con baja probabilidad de abandono."
        )

    st.metric(
        "Probabilidad de abandono",
        f"{prob_churn:.2%}"
    )

    st.progress(float(prob_churn))

# ==========================
# PIE DE PÁGINA
# ==========================

st.markdown("---")

st.caption(
    "Modelo de Machine Learning para predicción de churn "
    "implementado con Streamlit."
)