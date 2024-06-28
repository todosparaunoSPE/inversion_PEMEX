# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:57:43 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Título de la aplicación
st.title("Inversiones de PEMEX")

# Generar datos ficticios
np.random.seed(42)
years = np.arange(2010, 2024)
projects = ['Exploración', 'Producción', 'Refinación', 'Distribución', 'Otros']
data = {
    'Año': np.tile(years, len(projects)),
    'Proyecto': np.repeat(projects, len(years)),
    'Inversión (millones de USD)': np.random.randint(50, 500, len(years) * len(projects))
}

df = pd.DataFrame(data)

# Filtros de fecha
st.sidebar.subheader("Filtros de Fecha")
start_year, end_year = st.sidebar.select_slider("Selecciona un rango de años:",
                                                options=years, value=(years.min(), years.max()))

# Filtro por fecha
filtered_df = df[(df['Año'] >= start_year) & (df['Año'] <= end_year)]

# Comparación de proyectos
st.subheader("Comparación de Proyectos")
selected_projects = st.multiselect("Selecciona proyectos para comparar:", projects, default=projects[:2])

if selected_projects:
    filtered_df = filtered_df[filtered_df['Proyecto'].isin(selected_projects)]

# Mostrar el DataFrame filtrado
st.subheader("Datos de Inversiones Filtradas")
st.dataframe(filtered_df)

# Visualización interactiva con Plotly
fig = px.bar(filtered_df, x='Año', y='Inversión (millones de USD)', color='Proyecto', barmode='group',
             title="Inversiones de PEMEX por Proyecto y Año")
st.plotly_chart(fig)

# Gráfico de líneas para comparación de proyectos seleccionados
if selected_projects:
    fig3 = px.line(filtered_df, x='Año', y='Inversión (millones de USD)', color='Proyecto', 
                   title=f"Comparación de Inversiones en {', '.join(selected_projects)}")
    st.plotly_chart(fig3)

# Predicción de inversiones
st.subheader("Predicción de Inversiones")
selected_year = st.slider("Selecciona un año para predecir inversiones futuras:", 2025, 2030, 2025)

# Generar predicciones dinámicamente basadas en la selección del usuario usando regresión lineal
def predict_investments(df, selected_year):
    predictions = []
    for project in projects:
        project_df = df[df['Proyecto'] == project]
        X = project_df['Año'].values.reshape(-1, 1)
        y = project_df['Inversión (millones de USD)']
        model = LinearRegression().fit(X, y)
        predicted_investment = model.predict([[selected_year]])[0]
        predictions.append(predicted_investment)
    return predictions

predicted_investments = predict_investments(df, selected_year)
predictions = {
    'Año': [selected_year] * len(projects),
    'Proyecto': projects,
    'Inversión (millones de USD)': predicted_investments
}

pred_df = pd.DataFrame(predictions)

st.subheader(f"Predicciones para el año {selected_year}")
st.dataframe(pred_df)

fig4 = px.bar(pred_df, x='Proyecto', y='Inversión (millones de USD)', color='Proyecto',
              title=f"Predicción de Inversiones para el año {selected_year}")
st.plotly_chart(fig4)

# Análisis de correlación dinámico
st.subheader("Análisis de Correlación")
selected_corr_projects = st.multiselect("Selecciona proyectos para la matriz de correlación:", projects, default=projects)

if selected_corr_projects:
    corr_df = df[df['Proyecto'].isin(selected_corr_projects)].pivot(index='Año', columns='Proyecto', values='Inversión (millones de USD)')
    corr_matrix = corr_df.corr()
    fig5 = px.imshow(corr_matrix, title="Matriz de Correlación entre Proyectos", text_auto=True)
    st.plotly_chart(fig5)

# Exportación de datos
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(filtered_df)
st.sidebar.download_button(label="Descargar datos como CSV", data=csv, file_name='inversiones_pemex.csv', mime='text/csv')

# Análisis de tendencias
st.subheader("Análisis de Tendencias")
trend_project = st.selectbox("Selecciona un proyecto para analizar la tendencia:", projects)
trend_df = df[(df['Proyecto'] == trend_project) & (df['Año'] >= start_year) & (df['Año'] <= end_year)]
trend_fig = px.line(trend_df, x='Año', y='Inversión (millones de USD)', 
                    title=f"Tendencia de Inversiones en {trend_project}")
st.plotly_chart(trend_fig)

# Indicadores clave de desempeño (KPI)
st.sidebar.subheader("Indicadores Clave de Desempeño (KPI)")
total_investment = filtered_df['Inversión (millones de USD)'].sum()
average_investment = filtered_df['Inversión (millones de USD)'].mean()
max_investment = filtered_df['Inversión (millones de USD)'].max()
min_investment = filtered_df['Inversión (millones de USD)'].min()

st.sidebar.metric(label="Inversión Total (millones de USD)", value=f"{total_investment:.2f}")
st.sidebar.metric(label="Inversión Media (millones de USD)", value=f"{average_investment:.2f}")
st.sidebar.metric(label="Inversión Máxima (millones de USD)", value=f"{max_investment:.2f}")
st.sidebar.metric(label="Inversión Mínima (millones de USD)", value=f"{min_investment:.2f}")

# Información adicional en la barra lateral
st.sidebar.title("Ayuda")
st.sidebar.write("""
Esta aplicación muestra datos ficticios de inversiones de PEMEX en diferentes proyectos. 
Puedes ver la inversión total por año y proyecto, así como la evolución anual de la inversión en un proyecto específico.

### Funcionalidades:
- Comparación de inversiones entre múltiples proyectos.
- Predicción de inversiones futuras.
- Análisis de correlación entre proyectos.
- Filtros de fecha para seleccionar rangos de años específicos.
- Exportación de datos a CSV.
- Análisis de tendencias de inversión por proyecto.
- Indicadores clave de desempeño (KPI).

### Cómo usar:
1. Selecciona los proyectos para comparar.
2. Ajusta el año para ver las predicciones futuras.
3. Explora las correlaciones entre inversiones en diferentes proyectos.
4. Filtra los datos por rangos de fechas.
5. Descarga los datos en formato CSV para análisis adicionales.
6. Selecciona un proyecto para ver la tendencia de inversiones a lo largo del tiempo.
7. Consulta los indicadores clave de desempeño en la barra lateral.
""")

# Aviso de derechos de autor
st.sidebar.markdown("""
    ---
    © 2024. Todos los derechos reservados.
    Creado por jahoperi
""")   
