# Cargue de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import kagglehub
import os
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
import plotly.express as px
from scipy.stats import uniform

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import mca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Cirrosis Hepatica Streamlit App", layout="wide")
st.title("Clasificación de los estadios de la cirrosis hepática con métodos de Machine Learning")

st.caption("Estudio clínico de cirrosis hepática — ficha de variables")

texto = """
### **Variables:**

* **N_Days**: Número de días transcurridos entre el registro y la fecha más temprana entre fallecimiento, trasplante o análisis del estudio en 1986.  
* **Status**: estado del paciente C (censurado), CL (censurado por tratamiento hepático) o D (fallecimiento).  
* **Drug**: tipo de fármaco: D-penicilamina o placebo.  
* **Age**: edad en días.  
* **Sex**: M (hombre) o F (mujer).  
* **Ascites**: presencia de ascitis N (No) o Y (Sí).  
* **Hepatomegaly**: presencia de hepatomegalia N (No) o Y (Sí).  
* **Spiders**: presencia de aracnosis N (No) o Y (Sí).  
* **Edema**: presencia de edema N (sin edema ni tratamiento diurético), S (edema presente sin diuréticos o resuelto con diuréticos) o Y (edema a pesar del tratamiento diurético).  
* **Bilirubin**: bilirrubina sérica en mg/dl.  
* **Cholesterol**: colesterol sérico en mg/dl.  
* **Albumin**: albúmina en g/dl.  
* **Copper**: cobre en orina en µg/día.  
* **Alk_Phos**: fosfatasa alcalina en U/litro.  
* **SGOT**: SGOT en U/ml.  
* **Tryglicerides**: triglicéridos en mg/dl.  
* **Platelets**: plaquetas por metro cúbico [ml/1000].  
* **Prothrombin**: tiempo de protrombina en segundos.  
* **Stage**: estadio histológico de la enfermedad (1, 2 o 3).  

---

### **Dimensiones del dataset**
- **Tamaño:** 25 000 filas, 19 columnas  
- **Faltantes:** 0% en todas las columnas  

---
"""

st.markdown(texto)


# Descargar el dataset
path = kagglehub.dataset_download("aadarshvelu/liver-cirrhosis-stage-classification")
print("Ruta local del dataset:", path)

# Ver los archivos del dataset cargado
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

file_path = os.path.join(path, "liver_cirrhosis.csv")
df = pd.read_csv(file_path)

# Filtrar solo columnas categóricas (tipo "object" o "category")
cat_cols = df.select_dtypes(include=['object', 'category'])

st.subheader("Primeras 10 filas del dataset")
st.dataframe(df.head(10), use_container_width=True)

# ------- Helpers -------
def format_uniques(series, max_items=20):
    """Convierte valores únicos a una cadena legible, acota a max_items."""
    uniques = pd.Series(series.dropna().unique())
    head = uniques.head(max_items).astype(str).tolist()
    txt = ", ".join(head)
    if uniques.size > max_items:
        txt += f" … (+{uniques.size - max_items} más)"
    return txt

# ------- Detectar tipos -------
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# ------- Resumen variables categóricas -------
cat_summary = pd.DataFrame({
    "Variable": cat_cols,
    "Tipo de dato": [df[c].dtype for c in cat_cols],
    "Nº de categorías únicas": [df[c].nunique(dropna=True) for c in cat_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in cat_cols],
    "Categorías": [format_uniques(df[c], max_items=20) for c in cat_cols],
})

# ------- Resumen variables numéricas -------
num_summary = pd.DataFrame({
    "Variable": num_cols,
    "Tipo de dato": [df[c].dtype for c in num_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in num_cols],
    "Mínimo": [df[c].min(skipna=True) for c in num_cols],
    "Máximo": [df[c].max(skipna=True) for c in num_cols],
    "Media":  [df[c].mean(skipna=True) for c in num_cols],
    "Desviación estándar": [df[c].std(skipna=True) for c in num_cols],
}).round(2)

# ------- Mostrar en dos columnas iguales con separación uniforme -------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resumen variables categóricas")
    st.dataframe(cat_summary, use_container_width=True)

with col2:
    st.subheader("Resumen variables numéricas")
    st.dataframe(num_summary, use_container_width=True)





#####--------------------------------------------------------------------------------------#########

st.markdown("""### Análisis de variables categóricas""")
st.caption("Selecciona una variable para ver su distribución en tabla y gráfico de torta.")

variables_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if not variables_categoricas:
    st.warning("No se detectaron variables categóricas (object/category/bool) en df.")
    st.stop()

# =========================
# Controles (En la sección)
# =========================
st.markdown("*Controles*")
with st.container():
    c1, c2 = st.columns([1.6, 1.1])
    with c1:
        var = st.selectbox(
            "Variable categórica",
            options=variables_categoricas,
            index=0,
            key="cat_var_local"
        )
        top_n = st.slider(
            "Top N (agrupa el resto en 'Otros')",
            min_value=3, max_value=30, value=10, step=1,
            help="Agrupa las categorías menos frecuentes en 'Otros'",
            key="cat_topn_local"
        )
    with c2:
        incluir_na = st.checkbox("Incluir NaN", value=True, key="cat_incluir_na_local")
        orden_alfabetico = st.checkbox("Ordenar alfabéticamente (solo tabla)", value=False, key="cat_orden_local")

# =========================
# Preparar datos
# =========================
serie = df[var].copy()
if not incluir_na:
    serie = serie.dropna()

vc = serie.value_counts(dropna=incluir_na)

# Etiqueta amigable para NaN
labels = vc.index.to_list()
labels = ["(NaN)" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in labels]
counts = vc.values

data = pd.DataFrame({"Categoría": labels, "Conteo": counts})
data["Porcentaje"] = (data["Conteo"] / data["Conteo"].sum() * 100).round(2)

# Agrupar en "Otros" si supera Top N
if len(data) > top_n:
    top = data.nlargest(top_n, "Conteo").copy()
    otros = data.drop(top.index)
    fila_otros = pd.DataFrame({
        "Categoría": ["Otros"],
        "Conteo": [int(otros["Conteo"].sum())],
        "Porcentaje": [round(float(otros["Porcentaje"].sum()), 2)]
    })
    data_plot = pd.concat([top, fila_otros], ignore_index=True)
else:
    data_plot = data.copy()

# Orden por Conteo (siempre)
data_plot = data_plot.sort_values("Conteo", ascending=False).reset_index(drop=True)

# Orden opcional alfabético en la tabla (no afecta el gráfico)
data_table = data_plot.copy()
if orden_alfabetico:
    data_table = data_table.sort_values("Categoría").reset_index(drop=True)

# =========================
# Mostrar tabla y gráfico
# =========================
tcol, gcol = st.columns([1.1, 1.3], gap="large")

with tcol:
    st.subheader(f"Distribución de {var}")
    st.dataframe(
        data_table.assign(Porcentaje=data_table["Porcentaje"].round(2)),
        use_container_width=True
    )

with gcol:
    st.subheader("Gráfico de torta")
    chart = (
        alt.Chart(data_plot)
        .mark_arc(outerRadius=110)
        .encode(
            theta=alt.Theta(field="Conteo", type="quantitative"),
            color=alt.Color("Categoría:N", legend=alt.Legend(title="Categoría")),
            tooltip=[
                alt.Tooltip("Categoría:N"),
                alt.Tooltip("Conteo:Q", format=","),
                alt.Tooltip("Porcentaje:Q", format=".2f")
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Extras informativos
# =========================
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Categorías mostradas", f"{len(data_plot)}")
with c2:
    st.metric("Total registros (variable seleccionada)", f"{int(serie.shape[0]):,}".replace(",", "."))
with c3:
    st.metric("Incluye NaN", "Sí" if incluir_na else "No")

st.caption("Consejo: usa *Top N* para simplificar la lectura y agrupar categorías poco frecuentes en 'Otros'.")


#####--------------------------------------------------------------------------------------#########


# =========================
# Análisis de variables numéricas
# =========================
st.markdown("""### Análisis de variables numéricas""")
st.caption("Selecciona una variable para ver su distribución en tabla, boxplot e histograma.")

# Detectar variables numéricas
variables_numericas = df.select_dtypes(include=["number"]).columns.tolist()

if not variables_numericas:
    st.warning("No se detectaron variables numéricas en `df`.")
    st.stop()

# Controles (Sidebar)
st.sidebar.header("Controles - Numéricas")
var_num = st.sidebar.selectbox("Variable numérica", options=variables_numericas, index=0, key="num_var")
bins = st.sidebar.slider("Número de bins (histograma)", min_value=5, max_value=100, value=30, step=5)

# Preparar serie
serie_num = df[var_num].dropna()

# =========================
# Métricas descriptivas
# =========================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Nº datos no nulos", f"{serie_num.shape[0]:,}".replace(",", "."))
with c2:
    st.metric("Mínimo", f"{serie_num.min():.2f}")
with c3:
    st.metric("Máximo", f"{serie_num.max():.2f}")
with c4:
    st.metric("Media", f"{serie_num.mean():.2f}")
with c5:
    st.metric("Desv. Estándar", f"{serie_num.std():.2f}")

# =========================
# Gráficos
# =========================
g1, g2 = st.columns(2, gap="large")

with g1:
    st.subheader(f"Boxplot de `{var_num}`")
    box_data = pd.DataFrame({var_num: serie_num})
    box_chart = (
        alt.Chart(box_data)
        .mark_boxplot()
        .encode(y=alt.Y(var_num, type="quantitative"))
        .properties(height=300)
    )
    st.altair_chart(box_chart, use_container_width=True)

with g2:
    st.subheader(f"Histograma de `{var_num}`")
    hist_data = pd.DataFrame({var_num: serie_num})
    hist_chart = (
        alt.Chart(hist_data)
        .mark_bar()
        .encode(
            alt.X(var_num, bin=alt.Bin(maxbins=bins)),
            y='count()',
            tooltip=[alt.Tooltip(var_num, bin=alt.Bin(maxbins=bins)), alt.Tooltip('count()', title="Frecuencia")]
        )
        .properties(height=300)
    )
    st.altair_chart(hist_chart, use_container_width=True)


st.markdown("### Matriz de Correlación")

correlacion = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Matriz de Correlación")

st.pyplot(fig) 


# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 1. Selección de carácteristicas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.1. Selección de carácteristicas categóricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.2. Selección de carácteristicas numéricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.3. Unión de variables categóricas y númericas""")



# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 2. MCA Y PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.1. MCA""")

# split into train and test sets
df_cat = df.select_dtypes(include=['object', 'category'])
df_cat.info()

X = df_cat
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=1
)

# Codificación del conjunto de entrenamiento
X_train_encoded = pd.get_dummies(X_train)

# Codificación del conjunto de prueba
X_test_encoded = pd.get_dummies(X_test)

# Aplicar MCA
mca_cirrosis = mca.MCA(X_train_encoded, benzecri=True)

# Valores singulares y autovalores
sv = mca_cirrosis.s
eigvals = sv ** 2
explained_var = eigvals / eigvals.sum()
cum_explained_var = np.cumsum(explained_var)

# Graficar varianza acumulada
fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(range(1, len(cum_explained_var)+1), cum_explained_var, marker='o', linestyle='--')
ax1.axhline(y=0.8, color='r', linestyle='-')
ax1.set_xlabel('Dimensiones MCA')
ax1.set_ylabel('Varianza acumulada explicada')
ax1.set_title('Varianza acumulada explicada por MCA')
ax1.grid(True)
st.pyplot(fig1)

n_dims_90 = np.argmax(cum_explained_var >= 0.8) + 1  # +1 porque los índices empiezan en 0
st.write(f'Se necesitan {n_dims_90} dimensiones para explicar al menos el 80% de la varianza.')

# Coordenadas individuos (2 primeras dimensiones)
coords = mca_cirrosis.fs_r(N=3)

fig2, ax2 = plt.subplots(figsize=(8,6))
sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=y_train, palette='Set1', alpha=0.7, ax=ax2)
ax2.set_xlabel('Dimensión 1')
ax2.set_ylabel('Dimensión 2')
ax2.set_title('Scatterplot MCA Dim 1 vs Dim 2')
ax2.legend(title='Clase', labels=['Estadio 1', 'Estadio 2','Estadio 3'])
st.pyplot(fig2)

# Cargas variables categóricas (loadings) primeras 2 dimensiones
loadings_cat = pd.DataFrame(mca_cirrosis.fs_c()[:, :2], index=X_train_encoded.columns)

# Calcular contribución de cada variable (cuadrado / suma por dimensión)
loadings_sq = loadings_cat ** 2
contrib_cat = loadings_sq.div(loadings_sq.sum(axis=0), axis=1)

# Sumar contribuciones por variable
contrib_var = contrib_cat.sum(axis=1).sort_values(ascending=False)

fig3, ax3 = plt.subplots(figsize=(12,6))
contrib_var.plot(kind='bar', color='teal', ax=ax3)
ax3.set_ylabel('Contribución total a Dim 1 y 2')
ax3.set_title('Contribución de variables a las primeras 2 dimensiones MCA')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
fig3.tight_layout()
st.pyplot(fig3)

# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.2. PCA""")

df_num = df.select_dtypes(include=['int64', 'float64'])
df_num.info()

X = df_num
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=1
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# PCA con todos los componentes
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Varianza acumulada
explained_var = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='--')
ax.axhline(y=0.8, color='r', linestyle='-')
ax.set_xlabel('Número de componentes principales')
ax.set_ylabel('Varianza acumulada explicada')
ax.set_title('Varianza acumulada explicada por PCA')
ax.grid(True)
st.pyplot(fig)

n_dims_90 = np.argmax(explained_var >= 0.8) + 1
st.write(f'Se necesitan {n_dims_90} dimensiones para explicar al menos el 80% de la varianza.')

# Scatterplot PC1 vs PC2
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_train, palette='Set1', alpha=0.7, ax=ax2)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('Scatterplot PC1 vs PC2')
ax2.legend(title='Clase', labels=['Estadio 1', 'Estadio 2', 'Estadio 3'])
st.pyplot(fig2)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X_train.columns
)

st.write("X_pca shape:", X_pca.shape)
st.write("y_train shape:", y_train.shape)
st.write("Valores únicos en y_train:", y_train.unique())

#ráfico en 3D PCA

# PCA con 3 componentes
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Crear DataFrame con componentes principales y clases
df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'PC3': X_pca[:, 2],
    'Clase': y_train.values  # Asegura que es una columna alineada
})

# Mapear correctamente las clases 1, 2, 3
df_pca['Clase'] = df_pca['Clase'].astype(int).map({
    1: 'Estadio 1',
    2: 'Estadio 2',
    3: 'Estadio 3'
})

# Verifica que el DataFrame está bien (opcional para debug)
# st.write(df_pca.head())

# Crear gráfico interactivo 3D con Plotly
fig = px.scatter_3d(
    df_pca,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Clase',
    color_discrete_sequence=px.colors.qualitative.Set1,
    title='PCA 3D - Componentes Principales',
    labels={'Clase': 'Estadio'},
    opacity=0.7
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

fig3, ax3 = plt.subplots(figsize=(12,8))
sns.heatmap(loadings.iloc[:, :9], annot=True, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Heatmap de loadings (primeras 9 PCs)')
st.pyplot(fig3)

# PCA con componentes que explican al menos 80% varianza
pca_80 = PCA(n_components=0.80)
X_pca_80 = pca_80.fit_transform(X_scaled)

st.write(f"Número de componentes principales para explicar 80% varianza: {pca_80.n_components_}")
st.write(f"Varianza explicada acumulada por estas componentes: {sum(pca_80.explained_variance_ratio_)*100:.4f}%")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.3. Concatenar las dos matrices""")

# Datos numéricos

# Escalar variables numéricas del entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit solo al train

# Ajustar PCA al conjunto de entrenamiento
pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_scaled)

# Escalar el conjunto de prueba con el mismo scaler del entrenamiento
X_test_scaled = scaler.transform(X_test)  # sin fit

# Aplicar PCA ya entrenado al conjunto de prueba
X_test_pca = pca.transform(X_test_scaled)  # sin fit

# Datos categóricos

# Fit solo con entrenamiento
mca = prince.MCA(n_components=6, random_state=42)
mca = mca.fit(X_train_encoded)

# Transformación sobre entrenamiento y prueba
X_train_mca = mca.transform(X_train_encoded)
X_test_mca = mca.transform(X_test_encoded)

X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PCA_{i+1}' for i in range(X_train_pca.shape[1])])
X_train_mca_df = pd.DataFrame(X_train_mca.values, columns=[f'MCA_{i+1}' for i in range(X_train_mca.shape[1])])

X_train_final = pd.concat([X_train_pca_df, X_train_mca_df], axis=1)

# para el conjunto de prueba

X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PCA_{i+1}' for i in range(X_test_pca.shape[1])])
X_test_mca_df = pd.DataFrame(X_test_mca.values, columns=[f'MCA_{i+1}' for i in range(X_test_mca.shape[1])])

X_test_final = pd.concat([X_test_pca_df, X_test_mca_df], axis=1)

X_test_final.info()
st.subheader("Dataset final con variables PCA + MCA (Test Set)")
st.dataframe(X_test_final.head(10))  # Primeras 10 filas

# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.4. Modelado""")

models = {
    'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
}

resultados = []

for name, model in models.items():
    scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='accuracy')
    resultados.append({'Modelo': name, 'Accuracy promedio': scores.mean()})

df_resultados = pd.DataFrame(resultados)

st.subheader("Resultados de validación cruzada (accuracy promedio)")
st.table(df_resultados)
    
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.5. Ajuste de hiperparámetros""")

from sklearn.linear_model import LogisticRegression

param_dist = {
    'C': uniform(0.01, 10),
    'solver': ['lbfgs', 'saga'],
    'multi_class': ['multinomial']
}

log_reg = LogisticRegression(max_iter=1000)
random_log = RandomizedSearchCV(log_reg, param_distributions=param_dist, n_iter=20,
                                cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
random_log.fit(X_train_final, y_train)
print("Logistic Regression - Best Params:", random_log.best_params_)

from sklearn.neighbors import KNeighborsClassifier

param_dist = {
    'n_neighbors': randint(3, 20),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
random_knn = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=20,
                                cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
random_knn.fit(X_train_final, y_train)
print("KNN - Best Params:", random_knn.best_params_)

from sklearn.tree import DecisionTreeClassifier

param_dist = {
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'criterion': ['gini', 'entropy']
}

tree = DecisionTreeClassifier()
random_tree = RandomizedSearchCV(tree, param_distributions=param_dist, n_iter=20,
                                 cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
random_tree.fit(X_train_final, y_train)
print("Decision Tree - Best Params:", random_tree.best_params_)

from sklearn.ensemble import RandomForestClassifier

param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier()
random_rf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20,
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
random_rf.fit(X_train_final, y_train)
print("Random Forest - Best Params:", random_rf.best_params_)

from sklearn.svm import SVC

param_dist = {
    'C': uniform(0.1, 10),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm = SVC()
random_svm = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=20,
                                cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
random_svm.fit(X_train_final, y_train)
print("SVM - Best Params:", random_svm.best_params_)

st.subheader("Mejores hiperparámetros por modelo")

st.write("**Logistic Regression**")
st.write(random_log.best_params_)
st.write(f"Mejor accuracy (CV): {random_log.best_score_:.4f}")

st.write("**KNN**")
st.write(random_knn.best_params_)
st.write(f"Mejor accuracy (CV): {random_knn.best_score_:.4f}")

st.write("**Decision Tree**")
st.write(random_tree.best_params_)
st.write(f"Mejor accuracy (CV): {random_tree.best_score_:.4f}")

st.write("**Random Forest**")
st.write(random_rf.best_params_)
st.write(f"Mejor accuracy (CV): {random_rf.best_score_:.4f}")

# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.6. Comparación de modelos optimizados""")

modelos_optimizados = {
    "Logistic Regression": random_log.best_estimator_,
    "KNN": random_knn.best_estimator_,
    "Decision Tree": random_tree.best_estimator_,
    "Random Forest": random_rf.best_estimator_,
    "SVM": random_svm.best_estimator_
}

resultados = []

for nombre, modelo in modelos_optimizados.items():
    scores_cv = cross_val_score(modelo, X_train_final, y_train, cv=5, scoring='accuracy')
    mean_cv = scores_cv.mean()
    std_cv = scores_cv.std()

    modelo.fit(X_train_final, y_train)
    y_pred = modelo.predict(X_test_final)
    acc_test = accuracy_score(y_test, y_pred)

    resultados.append({
        'Modelo': nombre,
        'Accuracy CV (media)': round(mean_cv, 4),
        'Accuracy CV (std)': round(std_cv, 4),
        'Accuracy Test': round(acc_test, 4)
    })

    st.markdown(f"### 📌 Modelo: {nombre}")
    st.markdown(f"**Accuracy CV:** {mean_cv:.4f} ± {std_cv:.4f}")
    st.markdown(f"**Accuracy Test:** {acc_test:.4f}")
    st.text("📋 Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.text("🧩 Matriz de Confusión:")
    st.text(confusion_matrix(y_test, y_pred))

df_resultados = pd.DataFrame(resultados).sort_values(by='Accuracy Test', ascending=False)

st.markdown("## ✅ Resumen Comparativo de Modelos")
st.dataframe(df_resultados)
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 3. RFE""")
# ________________________________________________________________________________________________________________________________________________________________

