# Cargue de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# Instalar kagglehub y configurar Kaggle API
#!pip install kagglehub[pandas-datasets]

import kagglehub
import os
# Descargar el dataset
path = kagglehub.dataset_download("aadarshvelu/liver-cirrhosis-stage-classification")
print("Ruta local del dataset:", path)

# Ver los archivos del dataset cargado
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


file_path = os.path.join(path, "liver_cirrhosis.csv")
df = pd.read_csv(file_path)

df.info()


st.set_page_config(page_title="PCA Streamlit App", layout="wide")
st.title("PCA and MCA Analysis with NHANES Data")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/nhanes_2015_2016.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Asignar condiciones
def assign_condition(row):
    if (row["BPXSY1"] >= 140 or row["BPXSY2"] >= 140 or 
        row["BPXDI1"] >= 90 or row["BPXDI2"] >= 90):
        return "hypertension"
    elif row["BMXBMI"] >= 30:
        return "diabetes"
    elif ((row["RIAGENDR"] == 1 and row["BMXWAIST"] > 102) or 
          (row["RIAGENDR"] == 2 and row["BMXWAIST"] > 88)):
        return "high cholesterol"
    else:
        return "healthy"

df["Condition"] = df.apply(assign_condition, axis=1)

# Diccionario de códigos por variable categórica
category_mappings = {
    "RIAGENDR": {
        1: "Male",
        2: "Female"
    },
    "DMDMARTL": {
        1: "Married",
        2: "Divorced",
        3: "Never married",
        4: "Widowed",
        5: "Separated",
        6: "Living with partner",
        77: "Refused",
        99: "Don't know"
    },
    "DMDEDUC2": {
        1: "Less than 9th grade",
        2: "9-11th grade (no diploma)",
        3: "High school/GED",
        4: "Some college or AA degree",
        5: "College graduate or above",
        7: "Refused",
        9: "Don't know"
    },
    "SMQ020": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ101": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ110": {
        1: "Every day",
        2: "5–6 days/week",
        3: "3–4 days/week",
        4: "1–2 days/week",
        5: "2–3 days/month",
        6: "Once a month or less",
        7: "Refused",
        9: "Don't know"
    },
    "RIDRETH1": {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        5: "Other Race - Including Multi-Racial"
    },
    "DMDCITZN": {
        1: "Citizen by birth or naturalization",
        2: "Not a citizen of the U.S.",
        7: "Refused",
        9: "Don't know"
    },
    "HIQ210": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "SDMVPSU": {
        1: "PSU 1",
        2: "PSU 2"
    },
    "DMDHHSIZ": {
        1: "1 person",
        2: "2 people",
        3: "3 people",
        4: "4 people",
        5: "5 people",
        6: "6 people",
        7: "7 or more people"
    }
}

def apply_categorical_mappings(df, mappings):
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

df = apply_categorical_mappings(df, category_mappings)

col_map = {
    "SEQN": "Participant ID",
    "ALQ101": "Alcohol Intake - Past 12 months (Q1)",
    "ALQ110": "Alcohol Frequency",
    "ALQ130": "Alcohol Amount",
    "SMQ020": "Smoking Status",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age (years)",
    "RIDRETH1": "Race/Ethnicity",
    "DMDCITZN": "Citizenship",
    "DMDEDUC2": "Education Level",
    "DMDMARTL": "Marital Status",
    "DMDHHSIZ": "Household Size",
    "WTINT2YR": "Interview Weight",
    "SDMVPSU": "Masked PSU",
    "SDMVSTRA": "Masked Stratum",
    "INDFMPIR": "Income to Poverty Ratio",
    "BPXSY1": "Systolic BP1",
    "BPXDI1": "Diastolic BP1",
    "BPXSY2": "Systolic BP2",
    "BPXDI2": "Diastolic BP2",
    "BMXWT": "Body Weight",
    "BMXHT": "Body Height",
    "BMXBMI": "Body Mass Index",
    "BMXLEG": "Leg Length",
    "BMXARML": "Arm Length",
    "BMXARMC": "Arm Circumference",
    "BMXWAIST": "Waist Circumference",
    "HIQ210": "Health Insurance Coverage"
}

df = df.rename(columns=col_map)

# Asegurar compatibilidad con Arrow/Streamlit
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("string").fillna("Missing")

df = df.reset_index(drop=True)

# Mostrar info y variables categóricas lado a lado
st.subheader("Resumen de Datos")

# Crear columnas para mostrar info_df y category_df lado a lado
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tipo de Dato y Nulos**")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Dtype": df.dtypes.values
    })
    st.dataframe(info_df, use_container_width=True)

# Detección automática de variables categóricas
categorical_vars = [col for col in df.columns 
                    if df[col].dtype == 'object' or 
                       df[col].dtype == 'string' or 
                       df[col].nunique() <= 10]

with col2:
    st.markdown("**Variables Categóricas Detectadas**")
    category_info = []
    for col in categorical_vars:
        unique_vals = df[col].dropna().unique()
        category_info.append({
            "Variable": col,
            "Unique Classes": ", ".join(map(str, sorted(unique_vals)))
        })

    category_df = pd.DataFrame(category_info)
    st.dataframe(category_df, use_container_width=True)


st.subheader("Primeras 10 filas del dataset")
st.dataframe(df.head(10), use_container_width=True)

# Filtros
with st.sidebar:
    st.header("Filters")
    gender_filter = st.multiselect("Gender", sorted(df["Gender"].dropna().unique()))
    race_filter = st.multiselect("Race/Ethnicity", sorted(df["Race/Ethnicity"].dropna().unique()))
    condition_filter = st.multiselect("Condition", sorted(df["Condition"].dropna().unique()))
    #st.markdown("---")
    #k_vars = st.slider("Number of variables to select", 2, 10, 5)

# Aplicar filtros
for col, values in {
    "Gender": gender_filter, "Race/Ethnicity": race_filter, "Condition": condition_filter
}.items():
    if values:
        df = df[df[col].isin(values)]

if df.empty:
    st.warning("No data available after applying filters.")
    st.stop()

# Mostrar advertencias
problematic_cols = df.columns[df.dtypes == "object"].tolist()
nullable_ints = df.columns[df.dtypes.astype(str).str.contains("Int64")].tolist()

st.write("### ⚠️ Columnas potencialmente problemáticas para Arrow/Streamlit:")
if problematic_cols or nullable_ints:
    st.write("**Tipo 'object':**", problematic_cols)
    st.write("**Tipo 'Int64' (nullable):**", nullable_ints)
else:
    st.success("✅ No hay columnas problemáticas detectadas.")


# ==============================
# TRAIN / TEST SPLIT ANTES DE PCA y MCA
# ==============================
# 1️⃣ Eliminar filas con NaN en Condition
df = df.dropna(subset=["Condition"])

# --- 2. Separar X (features) y y (target) ---
X_df = df.drop(columns=["Condition"])
y_df = df["Condition"]

# --- 3. Identificar variables numéricas y categóricas ---
numeric_features = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# --- 4. Pipelines de imputación y escalado ---
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# --- 5. Preprocesador general ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# --- 6. Separar datos en train y test antes de PCA/MCA ---
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
)

# Mostrar info de división
st.write(f"Datos de entrenamiento: {X_train.shape[0]} filas")
st.write(f"Datos de prueba: {X_test.shape[0]} filas")

# --- 7. Transformar datos (sin PCA/MCA todavía) ---
# Ajustar el preprocesador
preprocessor.fit(X_train)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Obtener datos numéricos procesados
num_data_train = preprocessor.named_transformers_['num'].transform(X_train[numeric_features])
num_data_test = preprocessor.named_transformers_['num'].transform(X_test[numeric_features])

# Obtener datos categóricos procesados
cat_data_train = preprocessor.named_transformers_['cat'].transform(X_train[categorical_features])
cat_data_test = preprocessor.named_transformers_['cat'].transform(X_test[categorical_features])

# PCA sobre numéricas
from sklearn.decomposition import PCA
pca = PCA(n_components=7)
X_train_pca = pca.fit_transform(num_data_train)
X_test_pca = pca.transform(num_data_test)

pca = PCA(n_components=2)  # si quieres todas, usa n_components=None
X_pca = pca.fit_transform(X_train_processed)

# Crear DataFrame con resultados
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['condition'] = y_train.values  # color por la variable objetivo

# --- 1. Scatterplot PC1 vs PC2 ---
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='condition', palette='viridis', alpha=0.7)
plt.title('PCA - PC1 vs PC2')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()

## Obtener los loadings del PCA (componentes * características)
#loadings = X_pca.named_steps["pca"].components_

## Convertir a DataFrame con nombres de columnas
#loadings_df = pd.DataFrame(
#    loadings,
#    columns=X_num.columns,
#    index=[f"PC{i+1}" for i in range(loadings.shape[0])]
#).T  # Transponer para que columnas sean PCs y filas las variables

## Ordenar las filas por la importancia de la variable en la suma de cuadrados de los componentes
## Esto agrupa por aquellas variables con mayor contribución total
#loading_magnitude = (loadings_df**2).sum(axis=1)
#loadings_df["Importance"] = loading_magnitude
#loadings_df_sorted = loadings_df.sort_values(by="Importance", ascending=False).drop(columns="Importance")

## Graficar heatmap ordenado
#st.subheader("🔍 Heatmap de Loadings del PCA (Componentes Principales)")

#fig, ax = plt.subplots(figsize=(10, 12))
#sns.heatmap(loadings_df_sorted, annot=True, cmap="coolwarm", center=0, ax=ax)
#st.pyplot(fig)
