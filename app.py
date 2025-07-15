import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


# 1. Treinamento do modelo

df = pd.read_excel('dataframe.xlsx')

X = df[["bedrooms", "bathrooms", "age", "size"]]
y = df["price"]

model = LinearRegression().fit(X, y)
mean_price = y.mean()


# 3. Interface Streamlit

st.set_page_config(page_title="Predição de Imóveis", page_icon="🏠", layout="centered")
st.title("🏠 Predição de Valor de Imóvel")
st.write("Insira as características do imóvel para estimar seu valor de mercado.")

col1, col2 = st.columns(2)
with col1:
    bedrooms = st.slider("Número de quartos", min_value=1, max_value=5, value=3)
    bathrooms = st.slider("Número de banheiros", min_value=1, max_value=4, value=2)
with col2:
    age = st.slider("Idade do imóvel (anos)", min_value=0, max_value=50, value=10)
    size = st.slider("Tamanho do imóvel (m²)", min_value=40, max_value=250, value=100)

if st.button("🔮 Prever valor"):
    user_features = np.array([[bedrooms, bathrooms, age, size]])
    prediction = model.predict(user_features)[0]

    # Formatação em reais
    predicted_formatted = f"R$ {prediction:,.0f}".replace(",", ".")
    mean_formatted = f"R$ {mean_price:,.0f}".replace(",", ".")

    st.subheader("Resultado da predição")
    st.markdown(f"**Valor previsto:** {predicted_formatted}")


    # Plotly – gráfico de barras
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Valor Médio do Dataset",
        x=["Valor Médio"],
        y=[mean_price],
        text=[mean_formatted],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Valor Previsto",
        x=["Valor Previsto"],
        y=[prediction],
        text=[predicted_formatted],
        textposition="auto"
    ))

    fig.update_layout(
        title="Comparação entre Valor Previsto e Valor Médio do Dataset",
        yaxis_title="Preço (R$)",
        xaxis_title="Categoria",
        showlegend=False,
        bargap=0.4
    )

    st.plotly_chart(fig, use_container_width=True)


