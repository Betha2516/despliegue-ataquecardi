# ============================================================
#  Despliegue — Predicción de riesgo de ataque al corazón
#  Modelo: Random Forest (seleccionado en la fase de modelado)
# ============================================================
import json
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Riesgo Cardíaco · Random Forest",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Estilos (CSS)
# ------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f1116; }
    .hero {
        background: linear-gradient(120deg, #7f1d1d 0%, #b91c1c 45%, #ef4444 100%);
        padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(239,68,68,0.25);
    }
    .hero h1 { color: white; margin: 0; font-size: 1.9rem; }
    .hero p  { color: #fecaca; margin: .3rem 0 0 0; font-size: .95rem; }
    .card {
        background: #1a1d27; border: 1px solid #2a2e3a; border-radius: 16px;
        padding: 1.1rem 1.3rem; margin-bottom: 1rem;
    }
    .badge {
        display:inline-block; padding: .35rem .9rem; border-radius: 999px;
        font-weight:700; font-size: .95rem;
    }
    .badge-alto  { background:#7f1d1d; color:#fecaca; }
    .badge-medio { background:#78350f; color:#fde68a; }
    .badge-bajo  { background:#14532d; color:#bbf7d0; }
    .footer-note { color:#9ca3af; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Carga de artefactos (modelo, escalador, métricas, distribuciones)
# ------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    with open("modelo_rf_cardiaco.pkl", "rb") as f:
        modelo, labelencoder, variables, min_max_scaler = pickle.load(f)
    return modelo, labelencoder, list(variables), min_max_scaler


@st.cache_data
def cargar_metricas():
    with open("metricas_modelo.json") as f:
        return json.load(f)


@st.cache_data
def cargar_distribuciones():
    with open("distribuciones.json") as f:
        return json.load(f)


modelo, labelencoder, variables, min_max_scaler = cargar_modelo()
metricas = cargar_metricas()
dist = cargar_distribuciones()

VARIABLES_NUMERICAS = ["age", "avg_glucose_level"]

NOMBRES_AMIGABLES = {
    "age": "Edad",
    "avg_glucose_level": "Nivel de glucosa promedio",
    "smoking_status_'formerly smoked'": "Fumador anteriormente",
    "smoking_status_'never smoked'": "Nunca ha fumado",
    "smoking_status_Unknown": "Estado de fumador desconocido",
    "smoking_status_smokes": "Fumador actual",
    "hypertension_Yes": "Hipertensión",
    "heart_disease_Yes": "Enfermedad cardíaca previa",
    "ever_married_Yes": "Ha estado casado/a",
}

# Mapeo de opciones "bonitas" en la interfaz -> valor real que espera el modelo
# (en los datos originales, dos categorías de fumador quedaron con comillas literales)
OPCIONES_FUMADOR = {
    "Nunca ha fumado": "'never smoked'",
    "Fumador actualmente": "smokes",
    "Fumador anteriormente": "'formerly smoked'",
    "Desconocido / no reportado": "Unknown",
}

# ------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🫀 Predicción de riesgo de ataque al corazón</h1>
    <p>Modelo Random Forest entrenado sobre datos clínicos · Proyecto académico de clasificación</p>
</div>
""", unsafe_allow_html=True)

tab_prediccion, tab_desempeno = st.tabs(["🔍 Predicción individual", "📊 Desempeño del modelo"])

# ------------------------------------------------------------
# Sidebar — datos del paciente
# ------------------------------------------------------------
with st.sidebar:
    st.header("🧾 Datos del paciente")

    age = st.slider("Edad", min_value=0, max_value=85, value=45, step=1)
    avg_glucose_level = st.slider("Nivel de glucosa promedio (mg/dL)", min_value=50.0, max_value=280.0, value=100.0, step=0.5)

    st.divider()
    hypertension = st.radio("¿Tiene hipertensión?", ["No", "Sí"], horizontal=True)
    heart_disease = st.radio("¿Tiene antecedente de enfermedad cardíaca?", ["No", "Sí"], horizontal=True)
    ever_married = st.radio("¿Ha estado casado/a alguna vez?", ["No", "Sí"], horizontal=True)

    st.divider()
    smoking_label = st.selectbox("Estado de fumador", list(OPCIONES_FUMADOR.keys()))

    st.divider()
    predecir = st.button("💓 Calcular riesgo", use_container_width=True, type="primary")

    st.markdown(
        '<p class="footer-note">Ejercicio académico — no constituye diagnóstico médico. '
        'Ante cualquier síntoma real, consulte a un profesional de la salud.</p>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Preparación de una fila de entrada, igual que en el entrenamiento
# ------------------------------------------------------------
def preparar_entrada(age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status_valor):
    fila = pd.DataFrame([{
        "age": age,
        "hypertension": "Yes" if hypertension == "Sí" else "No",
        "heart_disease": "Yes" if heart_disease == "Sí" else "No",
        "ever_married": "Yes" if ever_married == "Sí" else "No",
        "avg_glucose_level": avg_glucose_level,
        "smoking_status": smoking_status_valor,
    }])

    fila_dummies = pd.get_dummies(fila, columns=["smoking_status"], drop_first=False, dtype=int)
    fila_dummies = pd.get_dummies(fila_dummies, columns=["hypertension", "heart_disease", "ever_married"], drop_first=True, dtype=int)

    # Alinea columnas exactamente como en el entrenamiento (agrega faltantes con 0)
    fila_final = fila_dummies.reindex(columns=variables, fill_value=0)
    fila_final[VARIABLES_NUMERICAS] = min_max_scaler.transform(fila_final[VARIABLES_NUMERICAS])
    return fila_final


def gauge_riesgo(probabilidad_pct: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad_pct,
        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": "#ef4444" if probabilidad_pct >= 50 else ("#f59e0b" if probabilidad_pct >= 20 else "#22c55e")},
            "bgcolor": "#1a1d27",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#14532d"},
                {"range": [20, 50], "color": "#78350f"},
                {"range": [50, 100], "color": "#7f1d1d"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10),
                       paper_bgcolor="#1a1d27", font={"color": "white"})
    return fig


def histograma_contexto(valor_usuario, datos_hist, titulo, color):
    edges = np.array(datos_hist["edges"])
    counts = np.array(datos_hist["counts"])
    centros = (edges[:-1] + edges[1:]) / 2
    fig = px.bar(x=centros, y=counts, labels={"x": titulo, "y": "N.º de pacientes"},
                 color_discrete_sequence=[color])
    fig.add_vline(x=valor_usuario, line_width=3, line_dash="dash", line_color="white",
                  annotation_text="Paciente actual", annotation_font_color="white")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                       paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
                       font={"color": "white"}, showlegend=False, title=titulo)
    return fig


def grafico_importancia():
    importancias = modelo.feature_importances_
    nombres = [NOMBRES_AMIGABLES.get(v, v) for v in variables]
    df_imp = pd.DataFrame({"variable": nombres, "importancia": importancias}).sort_values("importancia", ascending=True)
    fig = px.bar(df_imp, x="importancia", y="variable", orientation="h",
                 color="importancia", color_continuous_scale=["#f59e0b", "#ef4444"])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
                       font={"color": "white"}, coloraxis_showscale=False,
                       xaxis_title="Importancia media (Random Forest)", yaxis_title="")
    return fig


# ------------------------------------------------------------
# TAB 1 — Predicción individual
# ------------------------------------------------------------
with tab_prediccion:
    if not predecir:
        st.info("👈 Complete los datos del paciente en el panel izquierdo y presione **Calcular riesgo**.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(grafico_importancia(), use_container_width=True)
            st.caption("Importancia media de cada variable para el modelo Random Forest (calculada sobre todo el conjunto de datos).")
        with col_b:
            st.plotly_chart(histograma_contexto(dist["age_percentiles"]["50"], dist["age"], "Distribución de edad en los datos", "#60a5fa"), use_container_width=True)
    else:
        entrada = preparar_entrada(age, hypertension, heart_disease, ever_married,
                                    avg_glucose_level, OPCIONES_FUMADOR[smoking_label])
        pred = modelo.predict(entrada)[0]
        proba = modelo.predict_proba(entrada)[0]
        clase_pred = labelencoder.inverse_transform([pred])[0]
        idx_yes = list(labelencoder.classes_).index("Yes")
        prob_riesgo = proba[idx_yes] * 100

        if prob_riesgo >= 50:
            nivel, clase_css = "Riesgo alto", "badge-alto"
        elif prob_riesgo >= 20:
            nivel, clase_css = "Riesgo moderado", "badge-medio"
        else:
            nivel, clase_css = "Riesgo bajo", "badge-bajo"

        col1, col2, col3 = st.columns([1, 1, 1.4])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Predicción del modelo", "Riesgo de ataque" if clase_pred == "Yes" else "Sin riesgo detectado")
            st.markdown(f'<span class="badge {clase_css}">{nivel} · {prob_riesgo:.1f}%</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Resumen del paciente**")
            st.write(f"- Edad: **{age}** años (percentil ~{int(np.interp(age, list(dist['age_percentiles'].values()), [10,25,50,75,90])) if age else 0})")
            st.write(f"- Glucosa promedio: **{avg_glucose_level:.1f} mg/dL**")
            st.write(f"- Hipertensión: **{hypertension}** · Cardiopatía: **{heart_disease}**")
            st.write(f"- Estado civil (casado/a alguna vez): **{ever_married}** · Fumador: **{smoking_label}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.plotly_chart(gauge_riesgo(prob_riesgo), use_container_width=True)
            st.caption("Probabilidad estimada de la clase **Yes** (riesgo de ataque al corazón) según el Random Forest.")

        with col3:
            st.plotly_chart(grafico_importancia(), use_container_width=True)

        st.divider()
        colx, coly = st.columns(2)
        with colx:
            st.plotly_chart(histograma_contexto(age, dist["age"], "Edad del paciente vs. población de entrenamiento", "#60a5fa"), use_container_width=True)
        with coly:
            st.plotly_chart(histograma_contexto(avg_glucose_level, dist["avg_glucose_level"], "Glucosa del paciente vs. población de entrenamiento", "#f472b6"), use_container_width=True)

        st.warning("⚠️ Este resultado es producto de un modelo estadístico entrenado con fines académicos. "
                    "No sustituye una valoración médica profesional.")

# ------------------------------------------------------------
# TAB 2 — Desempeño del modelo
# ------------------------------------------------------------
with tab_desempeno:
    reporte = metricas["reporte"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitud (accuracy)", f"{reporte['accuracy']*100:.1f}%")
    c2.metric("Recall — clase Yes", f"{reporte['Yes']['recall']*100:.1f}%")
    c3.metric("Precisión — clase Yes", f"{reporte['Yes']['precision']*100:.1f}%")
    c4.metric("F1 — clase Yes", f"{reporte['Yes']['f1-score']*100:.1f}%")

    st.caption(f"Evaluado sobre el 30% de prueba · {metricas['n_estimators']} árboles · profundidad máxima {metricas['max_depth']}")

    df_reporte = pd.DataFrame(reporte).T.loc[["No", "Yes"], ["precision", "recall", "f1-score", "support"]]
    fig_bar = px.bar(df_reporte.reset_index().melt(id_vars="index", value_vars=["precision", "recall", "f1-score"]),
                      x="index", y="value", color="variable", barmode="group",
                      labels={"index": "Clase", "value": "Puntaje", "variable": "Métrica"},
                      color_discrete_sequence=["#60a5fa", "#ef4444", "#f59e0b"])
    fig_bar.update_layout(height=380, paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
                           font={"color": "white"}, yaxis_range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        f"El modelo se entrenó con balanceo sintético (SMOTE-NC) para la clase minoritaria "
        f"y `class_weight='balanced_subsample'`, priorizando **recall** en la clase *Yes* "
        f"(detectar el mayor número posible de casos de riesgo real), a costa de una menor precisión."
    )
