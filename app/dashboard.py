import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")
st.title("📊 Análisis de Sentimientos Multilingüe")

uploaded_file = st.file_uploader("📁 Sube un archivo CSV para analizar", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Datos cargados:")
    st.dataframe(df)

    if st.button("🔍 Analizar Sentimientos"):
        with st.spinner("Procesando el archivo..."):
            csv_buffer = uploaded_file.getvalue()

            response = requests.post(
                "http://127.0.0.1:8000/analyze-file/",
                files={"file": csv_buffer}
            )

        if response.status_code == 200:
            results = response.json()
            st.success("✅ Análisis completado con éxito")
            st.write("### 🧐 Vista previa de los resultados:")
            preview_df = pd.DataFrame(results["data_preview"])
            st.dataframe(preview_df)

            st.write("### 📊 Distribución de Sentimientos:")
            sentiment_counts = preview_df["sentiment_label"].value_counts()
            st.bar_chart(sentiment_counts)

            st.write("### ☁️ Nube de Palabras:")
            col1, col2 = st.columns([2, 1]) 

            with col1:
                all_text = " ".join(preview_df["cleaned_text"])
                wordcloud = WordCloud(
                    width=400, height=300, background_color="white"
                ).generate(all_text)
                fig_wc, ax_wc = plt.subplots(figsize=(4, 3))
                ax_wc.imshow(wordcloud, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

            with col2:
                st.write("### 👤 Porcentaje de Sentimientos:")
                user_sentiment = preview_df["sentiment_label"].value_counts(normalize=True) * 100
                st.write(user_sentiment)

        else:
            st.error(f"❌ Error al procesar el archivo: {response.json()['detail']}")

st.markdown("""
---
### ℹ️ Instrucciones:
1. **Sube un archivo CSV** que contenga los datos a analizar.
2. Los datos deben tener una columna `text` con el contenido a analizar.
3. Haz clic en **Analizar Sentimientos** para obtener los resultados y las visualizaciones.
""")
