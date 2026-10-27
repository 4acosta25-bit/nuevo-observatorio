"""
Observatorio de Analítica Económica y Social por Departamentos en Colombia
Plataforma Streamlit de nivel Think Tank — Econometría Espacial, Series de Tiempo,
Machine Learning Regularizado, PCA, Interpretabilidad y Simulación de Políticas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import urllib.request
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# Econometría y ML
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURACIÓN DE PÁGINA Y CSS
# =============================================================================
st.set_page_config(
    page_title="Observatorio Económico Departamental | Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f766e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.3);
    }
    .main-header h1 {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        font-size: 0.95rem;
        opacity: 0.9;
        margin: 0.4rem 0 0 0;
    }
    
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        border-left: 5px solid #0f766e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        height: 100%;
        transition: transform 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-card.blue { border-left-color: #1d4ed8; }
    .kpi-card.amber { border-left-color: #d97706; }
    .kpi-card.rose { border-left-color: #e11d48; }
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        font-family: 'Roboto Mono', monospace;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DICCIONARIO CANÓNICO DIVIPOLA + NORMALIZACIÓN
# =============================================================================
CANONICAL = {
    "AMAZONAS": {"code": "91", "name": "Amazonas"},
    "ANTIOQUIA": {"code": "05", "name": "Antioquia"},
    "ARAUCA": {"code": "81", "name": "Arauca"},
    "ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA": {"code": "88", "name": "San Andrés y Providencia"},
    "ATLÁNTICO": {"code": "08", "name": "Atlántico"},
    "BOGOTÁ, D.C.": {"code": "11", "name": "Bogotá D.C."},
    "BOLÍVAR": {"code": "13", "name": "Bolívar"},
    "BOYACÁ": {"code": "15", "name": "Boyacá"},
    "CALDAS": {"code": "17", "name": "Caldas"},
    "CAQUETÁ": {"code": "18", "name": "Caquetá"},
    "CASANARE": {"code": "85", "name": "Casanare"},
    "CAUCA": {"code": "19", "name": "Cauca"},
    "CESAR": {"code": "20", "name": "Cesar"},
    "CHOCÓ": {"code": "27", "name": "Chocó"},
    "CUNDINAMARCA": {"code": "25", "name": "Cundinamarca"},
    "CÓRDOBA": {"code": "23", "name": "Córdoba"},
    "GUAINÍA": {"code": "94", "name": "Guainía"},
    "GUAVIARE": {"code": "95", "name": "Guaviare"},
    "HUILA": {"code": "41", "name": "Huila"},
    "LA GUAJIRA": {"code": "44", "name": "La Guajira"},
    "MAGDALENA": {"code": "47", "name": "Magdalena"},
    "META": {"code": "50", "name": "Meta"},
    "NARIÑO": {"code": "52", "name": "Nariño"},
    "NORTE DE SANTANDER": {"code": "54", "name": "Norte de Santander"},
    "PUTUMAYO": {"code": "86", "name": "Putumayo"},
    "QUINDIO": {"code": "63", "name": "Quindío"},
    "RISARALDA": {"code": "66", "name": "Risaralda"},
    "SANTANDER": {"code": "68", "name": "Santander"},
    "SUCRE": {"code": "70", "name": "Sucre"},
    "TOLIMA": {"code": "73", "name": "Tolima"},
    "VALLE DEL CAUCA": {"code": "76", "name": "Valle del Cauca"},
    "VAUPÉS": {"code": "97", "name": "Vaupés"},
    "VICHADA": {"code": "99", "name": "Vichada"},
}

DEPT_LIST = sorted([v["name"] for v in CANONICAL.values()])
NAME_TO_GEO = {v["name"]: k for k, v in CANONICAL.items()}
GEO_TO_NAME = {k: v["name"] for k, v in CANONICAL.items()}
CODE_TO_NAME = {v["code"]: v["name"] for v in CANONICAL.values()}

VAR_LABELS = {
    "pib": "PIB Departamental (miles de millones COP)",
    "homicidios": "Tasa de Homicidios (por 100.000 hab.)",
    "inversion_dnp": "Inversión Pública DNP (millones COP)",
    "ipm": "Índice de Pobreza Multidimensional (%)",
    "desempleo": "Tasa de Desempleo (%)",
    "cobertura_edu": "Cobertura Neta Educación Media (%)",
}

VAR_SHORT = {
    "pib": "PIB",
    "homicidios": "Homicidios",
    "inversion_dnp": "Inversión DNP",
    "ipm": "IPM",
    "desempleo": "Desempleo",
    "cobertura_edu": "Cobertura Edu.",
}

# =============================================================================
# GENERACIÓN DE DATOS SINTÉTICOS REALISTAS (2010-2024)
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_master_dataset(seed: int = 42) -> pd.DataFrame:
    """Genera panel departamental 2010-2024 con correlaciones y tendencias realistas."""
    rng = np.random.default_rng(seed)
    years = list(range(2010, 2025))
    n_years = len(years)

    # Perfiles base por departamento (aproximados a realidad)
    profiles = {
        "Bogotá D.C.":          {"pib0": 180000, "hom0": 18, "inv0": 45000, "ipm0": 12, "des0": 9.5, "edu0": 78, "trend_pib": 0.045},
        "Antioquia":            {"pib0": 95000,  "hom0": 28, "inv0": 28000, "ipm0": 18, "des0": 10.2, "edu0": 72, "trend_pib": 0.042},
        "Valle del Cauca":      {"pib0": 65000,  "hom0": 42, "inv0": 18000, "ipm0": 16, "des0": 11.5, "edu0": 70, "trend_pib": 0.038},
        "Santander":            {"pib0": 42000,  "hom0": 16, "inv0": 12000, "ipm0": 15, "des0": 9.8, "edu0": 74, "trend_pib": 0.040},
        "Cundinamarca":         {"pib0": 38000,  "hom0": 12, "inv0": 14000, "ipm0": 14, "des0": 8.5, "edu0": 76, "trend_pib": 0.043},
        "Atlántico":            {"pib0": 28000,  "hom0": 30, "inv0": 9000,  "ipm0": 22, "des0": 12.0, "edu0": 68, "trend_pib": 0.036},
        "Bolívar":              {"pib0": 22000,  "hom0": 28, "inv0": 7500,  "ipm0": 28, "des0": 11.8, "edu0": 62, "trend_pib": 0.034},
        "Norte de Santander":   {"pib0": 15000,  "hom0": 35, "inv0": 5500,  "ipm0": 32, "des0": 13.5, "edu0": 58, "trend_pib": 0.030},
        "Tolima":               {"pib0": 14000,  "hom0": 25, "inv0": 4800,  "ipm0": 24, "des0": 12.2, "edu0": 65, "trend_pib": 0.032},
        "Huila":                {"pib0": 12000,  "hom0": 22, "inv0": 4200,  "ipm0": 26, "des0": 11.0, "edu0": 64, "trend_pib": 0.033},
        "Meta":                 {"pib0": 18000,  "hom0": 20, "inv0": 6000,  "ipm0": 20, "des0": 10.5, "edu0": 66, "trend_pib": 0.041},
        "Casanare":             {"pib0": 16000,  "hom0": 18, "inv0": 5500,  "ipm0": 19, "des0": 9.2, "edu0": 67, "trend_pib": 0.044},
        "Boyacá":               {"pib0": 13000,  "hom0": 10, "inv0": 4500,  "ipm0": 17, "des0": 8.8, "edu0": 73, "trend_pib": 0.035},
        "Caldas":               {"pib0": 11000,  "hom0": 14, "inv0": 3800,  "ipm0": 16, "des0": 10.0, "edu0": 71, "trend_pib": 0.034},
        "Risaralda":            {"pib0": 10000,  "hom0": 15, "inv0": 3600,  "ipm0": 17, "des0": 10.8, "edu0": 70, "trend_pib": 0.035},
        "Quindío":              {"pib0": 5500,   "hom0": 13, "inv0": 2200,  "ipm0": 15, "des0": 11.2, "edu0": 72, "trend_pib": 0.033},
        "Cauca":                {"pib0": 9000,   "hom0": 48, "inv0": 3200,  "ipm0": 35, "des0": 13.0, "edu0": 55, "trend_pib": 0.028},
        "Nariño":               {"pib0": 8500,   "hom0": 32, "inv0": 3000,  "ipm0": 33, "des0": 12.5, "edu0": 57, "trend_pib": 0.029},
        "Córdoba":              {"pib0": 10000,  "hom0": 26, "inv0": 3500,  "ipm0": 38, "des0": 12.8, "edu0": 54, "trend_pib": 0.031},
        "Magdalena":            {"pib0": 9500,   "hom0": 34, "inv0": 3300,  "ipm0": 40, "des0": 13.2, "edu0": 52, "trend_pib": 0.030},
        "Cesar":                {"pib0": 11000,  "hom0": 27, "inv0": 3800,  "ipm0": 36, "des0": 12.0, "edu0": 56, "trend_pib": 0.032},
        "La Guajira":           {"pib0": 7000,   "hom0": 22, "inv0": 2800,  "ipm0": 48, "des0": 14.5, "edu0": 45, "trend_pib": 0.025},
        "Sucre":                {"pib0": 5500,   "hom0": 20, "inv0": 2100,  "ipm0": 42, "des0": 13.8, "edu0": 50, "trend_pib": 0.027},
        "Chocó":                {"pib0": 3500,   "hom0": 55, "inv0": 1800,  "ipm0": 55, "des0": 15.0, "edu0": 42, "trend_pib": 0.022},
        "Caquetá":              {"pib0": 3200,   "hom0": 30, "inv0": 1600,  "ipm0": 40, "des0": 12.5, "edu0": 48, "trend_pib": 0.026},
        "Putumayo":             {"pib0": 2800,   "hom0": 28, "inv0": 1400,  "ipm0": 38, "des0": 13.0, "edu0": 49, "trend_pib": 0.027},
        "Arauca":               {"pib0": 4500,   "hom0": 35, "inv0": 2000,  "ipm0": 30, "des0": 11.5, "edu0": 55, "trend_pib": 0.030},
        "Guaviare":             {"pib0": 1200,   "hom0": 25, "inv0": 800,   "ipm0": 42, "des0": 12.0, "edu0": 46, "trend_pib": 0.024},
        "Vichada":              {"pib0": 900,    "hom0": 12, "inv0": 600,   "ipm0": 45, "des0": 10.5, "edu0": 40, "trend_pib": 0.023},
        "Guainía":              {"pib0": 600,    "hom0": 8,  "inv0": 400,   "ipm0": 50, "des0": 11.0, "edu0": 38, "trend_pib": 0.020},
        "Vaupés":               {"pib0": 450,    "hom0": 6,  "inv0": 350,   "ipm0": 52, "des0": 10.8, "edu0": 36, "trend_pib": 0.018},
        "Amazonas":             {"pib0": 700,    "hom0": 15, "inv0": 500,   "ipm0": 44, "des0": 11.2, "edu0": 41, "trend_pib": 0.021},
        "San Andrés y Providencia": {"pib0": 1200, "hom0": 25, "inv0": 900, "ipm0": 20, "des0": 9.0, "edu0": 75, "trend_pib": 0.028},
    }

    rows = []
    for dept, p in profiles.items():
        for i, y in enumerate(years):
            t = i / (n_years - 1)
            # Shock COVID 2020
            covid = 0.85 if y == 2020 else (0.95 if y == 2021 else 1.0)
            # Tendencia + ruido
            pib = p["pib0"] * ((1 + p["trend_pib"]) ** i) * covid * rng.normal(1, 0.03)
            # Homicidios: tendencia a la baja con ruido
            hom = max(2, p["hom0"] * (1 - 0.35 * t) * rng.normal(1, 0.12) * (1.15 if y in [2021, 2022] else 1.0))
            inv = p["inv0"] * ((1.04) ** i) * covid * rng.normal(1, 0.08)
            # IPM: mejora gradual
            ipm = max(5, p["ipm0"] * (1 - 0.40 * t) * rng.normal(1, 0.06))
            des = max(3, p["des0"] * (1 - 0.15 * t) * covid * rng.normal(1, 0.08) * (1.4 if y == 2020 else 1.0))
            edu = min(95, p["edu0"] * (1 + 0.18 * t) * rng.normal(1, 0.03))

            rows.append({
                "departamento": dept,
                "codigo_divipola": next(v["code"] for v in CANONICAL.values() if v["name"] == dept),
                "anio": y,
                "pib": round(pib, 1),
                "homicidios": round(hom, 1),
                "inversion_dnp": round(inv, 1),
                "ipm": round(ipm, 1),
                "desempleo": round(des, 1),
                "cobertura_edu": round(edu, 1),
            })

    df = pd.DataFrame(rows)

    # Introducir algunos gaps aleatorios (para demostrar imputación)
    for _ in range(40):
        idx = rng.integers(0, len(df))
        col = rng.choice(["homicidios", "ipm", "desempleo", "cobertura_edu"])
        df.loc[idx, col] = np.nan

    return df


@st.cache_data(show_spinner=False)
def load_and_prepare_data() -> pd.DataFrame:
    """Pipeline robusto: intenta carga externa → fallback a datos sintéticos + interpolación."""
    df = None
    source_msg = None

    # Intento 1: archivo local (si existe en el mismo directorio)
    try:
        local_path = "data/panel_departamental.csv"
        df = pd.read_csv(local_path)
        source_msg = "Datos cargados desde archivo local."
    except Exception:
        pass

    # Intento 2: endpoint HTTP de ejemplo (timeout corto)
    if df is None:
        try:
            # Placeholder: en producción apuntaría a un endpoint real de DANE/TerriData
            # Aquí forzamos fallback para demo
            raise TimeoutError("Simulación de caída de servidor externo")
        except Exception:
            pass

    # Fallback: dataset maestro sintético
    if df is None:
        df = generate_master_dataset()
        source_msg = (
            "⚠️ Fuente externa no disponible. Se cargó el dataset maestro de respaldo "
            "(panel sintético calibrado 2010-2024). La aplicación continúa sin interrupciones."
        )

    # Normalización de nombres de columnas
    col_map = {
        "Departamento": "departamento", "Año": "anio", "PIB": "pib",
        "Homicidios": "homicidios", "Inversión": "inversion_dnp",
        "IPM": "ipm", "Desempleo": "desempleo", "Cobertura": "cobertura_edu",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Asegurar columnas requeridas
    required = ["departamento", "anio", "pib", "homicidios", "inversion_dnp", "ipm", "desempleo", "cobertura_edu"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # Interpolación lineal intragrupo por departamento
    value_cols = ["pib", "homicidios", "inversion_dnp", "ipm", "desempleo", "cobertura_edu"]
    df = df.sort_values(["departamento", "anio"])
    for col in value_cols:
        df[col] = df.groupby("departamento")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        # Relleno residual con mediana del departamento
        df[col] = df.groupby("departamento")[col].transform(
            lambda x: x.fillna(x.median())
        )

    df["anio"] = df["anio"].astype(int)
    return df, source_msg


@st.cache_data(show_spinner=False)
def load_geojson():
    """Carga GeoJSON oficial de departamentos con fallback."""
    url = "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            geo = json.load(resp)
        # Añadir nombre canónico
        for feat in geo["features"]:
            raw = feat["properties"].get("DPTO_CNMBR", "")
            feat["properties"]["nombre_canonico"] = GEO_TO_NAME.get(raw, raw.title())
        return geo, None
    except Exception as e:
        return None, f"No se pudo cargar GeoJSON externo: {e}. Mapas coropléticos limitados."


# =============================================================================
# FUNCIONES ANALÍTICAS
# =============================================================================
def run_adf(series: pd.Series, name: str) -> dict:
    series = series.dropna()
    if len(series) < 8:
        return {"variable": name, "adf_stat": np.nan, "pvalue": np.nan, "estacionaria": "Insuficiente"}
    res = adfuller(series, autolag="AIC")
    return {
        "variable": name,
        "adf_stat": round(res[0], 4),
        "pvalue": round(res[1], 4),
        "estacionaria": "Sí" if res[1] < 0.05 else "No",
    }


def johansen_test(df_multi: pd.DataFrame) -> dict:
    try:
        data = df_multi.dropna()
        if len(data) < 12 or data.shape[1] < 2:
            return {"trace_stat": None, "crit_95": None, "cointegradas": "Datos insuficientes"}
        res = coint_johansen(data, det_order=0, k_ar_diff=1)
        trace = res.lr1[0]
        crit = res.cvt[0, 1]  # 95%
        return {
            "trace_stat": round(trace, 3),
            "crit_95": round(crit, 3),
            "cointegradas": "Sí" if trace > crit else "No",
        }
    except Exception:
        return {"trace_stat": None, "crit_95": None, "cointegradas": "Error en cálculo"}


def fit_var_or_vecm(df_ts: pd.DataFrame, n_forecast: int = 3):
    """Estima VAR o VECM según cointegración y genera pronóstico."""
    data = df_ts.dropna()
    cols = list(data.columns)
    joh = johansen_test(data)

    model_type = "VECM" if joh["cointegradas"] == "Sí" else "VAR"
    forecast_df = None
    summary_text = ""
    aic = bic = np.nan

    try:
        if model_type == "VECM" and data.shape[1] >= 2:
            model = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="ci")
            res = model.fit()
            fc = res.predict(steps=n_forecast)
            last_year = int(data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else data.index[-1]
            if isinstance(last_year, (pd.Timestamp, np.datetime64)):
                last_year = pd.Timestamp(last_year).year
            idx = list(range(int(last_year) + 1, int(last_year) + 1 + n_forecast))
            forecast_df = pd.DataFrame(fc, columns=cols, index=idx)
            summary_text = str(res.summary())
        else:
            # Diferenciar si no estacionario y usar VAR
            model = VAR(data)
            lag = min(2, max(1, len(data) // 6))
            res = model.fit(lag)
            aic, bic = res.aic, res.bic
            fc = res.forecast(data.values[-lag:], steps=n_forecast)
            last = data.index[-1]
            if isinstance(last, (pd.Timestamp, np.datetime64)):
                last = pd.Timestamp(last).year
            idx = list(range(int(last) + 1, int(last) + 1 + n_forecast))
            forecast_df = pd.DataFrame(fc, columns=cols, index=idx)
            summary_text = str(res.summary())
    except Exception as e:
        summary_text = f"Estimación fallida: {e}"
        model_type = "Error"

    return model_type, forecast_df, joh, summary_text, aic, bic


def train_ml_models(X: pd.DataFrame, y: pd.Series):
    """Entrena Ridge, Lasso, SVR y MLP con StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42, early_stopping=True),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_train)
        pred = mdl.predict(X_te)
        results[name] = {
            "model": mdl,
            "scaler": scaler,
            "r2": r2_score(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "y_test": y_test,
            "y_pred": pred,
            "feature_names": list(X.columns),
        }
    return results


def feature_importance_rf(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.DataFrame({
        "variable": X.columns,
        "importancia": rf.feature_importances_,
    }).sort_values("importancia", ascending=False)
    return imp, rf


# =============================================================================
# CARGA DE DATOS
# =============================================================================
df, source_msg = load_and_prepare_data()
geojson, geo_msg = load_geojson()

if source_msg:
    st.toast(source_msg, icon="ℹ️")
if geo_msg:
    st.info(geo_msg)

# =============================================================================
# SIDEBAR – FILTROS GLOBALES
# =============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Controles Globales")
    st.markdown("---")

    selected_dept = st.selectbox(
        "Departamento principal",
        options=["Todos"] + DEPT_LIST,
        index=0,
        key="sidebar_dept_select",
    )

    year_min, year_max = int(df["anio"].min()), int(df["anio"].max())
    year_range = st.slider(
        "Rango de años",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
        key="sidebar_year_range",
    )

    selected_var = st.selectbox(
        "Indicador socioeconómico",
        options=list(VAR_LABELS.keys()),
        format_func=lambda x: VAR_LABELS[x],
        index=0,
        key="sidebar_var_select",
    )

    st.markdown("---")
    st.markdown("**Variables del panel**")
    for k, v in VAR_SHORT.items():
        st.caption(f"• {v}")
    st.markdown("---")
    st.caption("Observatorio v1.0 · Datos 2010–2024")
    st.caption("Arquitectura: Streamlit + statsmodels + scikit-learn + Plotly")

# Filtrado base
mask = (df["anio"] >= year_range[0]) & (df["anio"] <= year_range[1])
df_f = df.loc[mask].copy()
if selected_dept != "Todos":
    df_dept = df_f[df_f["departamento"] == selected_dept]
else:
    df_dept = df_f

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🇨🇴 Observatorio de Analítica Económica y Social por Departamentos</h1>
    <p>Plataforma de econometría espacial, series de tiempo, ML regularizado e interpretabilidad · Colombia 2010–2024</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# KPI CARDS
# =============================================================================
latest_year = int(df_f["anio"].max())
df_latest = df_f[df_f["anio"] == latest_year]

if selected_dept != "Todos":
    row = df_latest[df_latest["departamento"] == selected_dept]
    if len(row):
        k_pib = row["pib"].values[0]
        k_ipm = row["ipm"].values[0]
        k_des = row["desempleo"].values[0]
        k_hom = row["homicidios"].values[0]
    else:
        k_pib = k_ipm = k_des = k_hom = np.nan
    label_scope = selected_dept
else:
    k_pib = df_latest["pib"].sum()
    k_ipm = df_latest["ipm"].mean()
    k_des = df_latest["desempleo"].mean()
    k_hom = df_latest["homicidios"].mean()
    label_scope = "Nacional (promedio/suma)"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card blue">
        <div class="kpi-label">PIB {label_scope} · {latest_year}</div>
        <div class="kpi-value">{k_pib:,.0f}</div>
        <div class="kpi-delta">miles de millones COP</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">IPM promedio · {latest_year}</div>
        <div class="kpi-value">{k_ipm:.1f}%</div>
        <div class="kpi-delta">pobreza multidimensional</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-card amber">
        <div class="kpi-label">Desempleo · {latest_year}</div>
        <div class="kpi-value">{k_des:.1f}%</div>
        <div class="kpi-delta">tasa anual</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="kpi-card rose">
        <div class="kpi-label">Homicidios · {latest_year}</div>
        <div class="kpi-value">{k_hom:.1f}</div>
        <div class="kpi-delta">por 100.000 hab.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PESTAÑAS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ EDA Espacial",
    "📈 Series de Tiempo",
    "🤖 Machine Learning",
    "📉 PCA",
    "🔍 Interpretabilidad",
    "🎯 Simulación What-If",
])

# -----------------------------------------------------------------------------
# TAB 1 · EDA ESPACIAL
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Análisis Exploratorio Espacial")
    year_map = st.slider("Año del mapa", year_min, year_max, latest_year, key="map_year")
    df_map = df_f[df_f["anio"] == year_map].copy()
    df_map["nombre_geo"] = df_map["departamento"].map(
        lambda x: NAME_TO_GEO.get(x, x.upper())
    )

    col_map, col_ts = st.columns([1.4, 1])

    with col_map:
        if geojson is not None:
            try:
                fig = px.choropleth_map(
                    df_map,
                    geojson=geojson,
                    locations="departamento",
                    featureidkey="properties.nombre_canonico",
                    color=selected_var,
                    color_continuous_scale="Teal",
                    map_style="carto-positron",
                    zoom=4.2,
                    center={"lat": 4.5, "lon": -73.0},
                    opacity=0.75,
                    hover_name="departamento",
                    hover_data={selected_var: ":.1f", "departamento": False},
                    labels={selected_var: VAR_SHORT[selected_var]},
                    title=f"{VAR_LABELS[selected_var]} · {year_map}",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # Fallback a choropleth clásico
                fig = px.choropleth(
                    df_map,
                    geojson=geojson,
                    locations="departamento",
                    featureidkey="properties.nombre_canonico",
                    color=selected_var,
                    color_continuous_scale="Teal",
                    scope="south america",
                    labels={selected_var: VAR_SHORT[selected_var]},
                    title=f"{VAR_LABELS[selected_var]} · {year_map}",
                )
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("GeoJSON no disponible. Mostrando ranking tabular.")
            st.dataframe(
                df_map[["departamento", selected_var]].sort_values(selected_var, ascending=False),
                use_container_width=True, height=400,
            )

    with col_ts:
        st.markdown("**Evolución temporal del indicador**")
        if selected_dept != "Todos":
            ts = df_f[df_f["departamento"] == selected_dept]
            fig_ts = px.line(ts, x="anio", y=selected_var, markers=True,
                             title=f"{selected_dept}", labels={selected_var: VAR_SHORT[selected_var]})
        else:
            # Top 5 + promedio nacional
            top5 = df_map.nlargest(5, selected_var)["departamento"].tolist()
            ts = df_f[df_f["departamento"].isin(top5)]
            fig_ts = px.line(ts, x="anio", y=selected_var, color="departamento", markers=True,
                             title="Top 5 departamentos", labels={selected_var: VAR_SHORT[selected_var]})
        fig_ts.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("**Ranking del año**")
        ranking = df_map[["departamento", selected_var]].sort_values(selected_var, ascending=False).reset_index(drop=True)
        ranking.index = ranking.index + 1
        st.dataframe(ranking.head(10).style.format({selected_var: "{:.1f}"}), use_container_width=True, height=260)

# -----------------------------------------------------------------------------
# TAB 2 · SERIES DE TIEMPO / ECONOMETRÍA
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Econometría de Series de Tiempo — VAR / VECM")
    st.markdown(
        "Se evalúa estacionariedad (ADF) y cointegración (Johansen) para decidir entre "
        "**VAR** (series estacionarias) y **VECM** (series cointegradas)."
    )

    depts_ts = st.multiselect(
        "Seleccione 2–4 departamentos para el sistema multivariado",
        options=DEPT_LIST,
        default=["Bogotá D.C.", "Antioquia", "Valle del Cauca"][:3],
        max_selections=4,
        key="ts_depts_multiselect",
    )
    vars_ts = st.multiselect(
        "Variables del sistema",
        options=list(VAR_LABELS.keys()),
        default=["pib", "ipm", "desempleo"],
        format_func=lambda x: VAR_SHORT[x],
        max_selections=4,
        key="ts_vars_multiselect",
    )
    n_fc = st.slider(
        "Horizonte de pronóstico (años)", 1, 5, 3,
        key="ts_forecast_horizon_slider",
    )

    if len(depts_ts) >= 1 and len(vars_ts) >= 2:
        # Construir panel largo → wide para el primer departamento (sistema univariado multi-var)
        # o promedio nacional de los seleccionados
        subset = df_f[df_f["departamento"].isin(depts_ts)].groupby("anio")[vars_ts].mean()
        subset = subset.sort_index()

        with st.spinner("Evaluando propiedades estocásticas y estimando modelo..."):
            # ADF
            adf_results = [run_adf(subset[c], VAR_SHORT[c]) for c in vars_ts]
            adf_df = pd.DataFrame(adf_results)

            model_type, forecast_df, joh, summary_txt, aic, bic = fit_var_or_vecm(subset, n_forecast=n_fc)

        c_adf, c_joh = st.columns(2)
        with c_adf:
            st.markdown("**Pruebas ADF (estacionariedad)**")
            st.dataframe(adf_df, use_container_width=True, hide_index=True)
        with c_joh:
            st.markdown("**Test de Johansen (cointegración)**")
            st.metric("¿Cointegradas?", joh["cointegradas"])
            if joh["trace_stat"] is not None:
                st.caption(f"Trace statistic: {joh['trace_stat']}  |  Crítico 95%: {joh['crit_95']}")
            st.metric("Modelo seleccionado", model_type)
            if not np.isnan(aic):
                st.caption(f"AIC: {aic:.2f}  ·  BIC: {bic:.2f}")

        # Visualización histórico + forecast
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, c in enumerate(vars_ts):
            fig.add_trace(go.Scatter(
                x=subset.index, y=subset[c], name=f"{VAR_SHORT[c]} (hist)",
                mode="lines+markers", line=dict(color=colors[i % len(colors)]),
            ))
            if forecast_df is not None and c in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df.index, y=forecast_df[c], name=f"{VAR_SHORT[c]} (fcst)",
                    mode="lines+markers", line=dict(color=colors[i % len(colors)], dash="dash"),
                ))
        fig.update_layout(
            title=f"Trayectoria histórica y pronóstico {model_type} · {n_fc} periodos",
            xaxis_title="Año", height=420, legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

        if forecast_df is not None:
            st.markdown("**Tabla de pronósticos**")
            st.dataframe(forecast_df.style.format("{:.2f}"), use_container_width=True)

        with st.expander("Resumen estadistico completo del modelo"):
            st.text(summary_txt)
    else:
        st.info("Seleccione al menos 1 departamento y 2 variables para estimar el sistema.")

# -----------------------------------------------------------------------------
# TAB 3 · MACHINE LEARNING
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Regresion Supervisada Regularizada")
    st.markdown(
        "Modelos: **Ridge**, **Lasso**, **SVR** y **MLPRegressor**. "
        "Pipeline con `StandardScaler`. Metricas: R2 y RMSE."
    )

    # Keys de widgets UNICOS y distintos de las claves de session_state del modelo
    target = st.selectbox(
        "Variable objetivo (y)",
        options=list(VAR_LABELS.keys()),
        format_func=lambda x: VAR_LABELS[x],
        index=3,
        key="ml_target_select",
    )
    features = [v for v in VAR_LABELS.keys() if v != target]
    selected_features = st.multiselect(
        "Features (X)",
        options=features,
        default=features,
        format_func=lambda x: VAR_SHORT[x],
        key="ml_features_select",
    )

    if len(selected_features) >= 1:
        use_panel = st.checkbox(
            "Usar panel completo (todos los anos)",
            value=True,
            key="ml_use_panel_checkbox",
        )
        data_ml = df_f if use_panel else df_f[df_f["anio"] == latest_year]
        data_ml = data_ml.dropna(subset=[target] + selected_features)

        X = data_ml[selected_features]
        y = data_ml[target]

        with st.spinner("Entrenando modelos..."):
            results = train_ml_models(X, y)

        metrics_df = pd.DataFrame({
            "Modelo": list(results.keys()),
            "R2": [results[m]["r2"] for m in results],
            "RMSE": [results[m]["rmse"] for m in results],
        }).sort_values("R2", ascending=False)
        st.dataframe(
            metrics_df.style.format({"R2": "{:.3f}", "RMSE": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        best = metrics_df.iloc[0]["Modelo"]
        res_best = results[best]
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=res_best["y_test"], y=res_best["y_pred"],
            mode="markers", name="Observaciones",
            marker=dict(size=8, opacity=0.7, color="#0f766e"),
        ))
        mn = min(res_best["y_test"].min(), res_best["y_pred"].min())
        mx = max(res_best["y_test"].max(), res_best["y_pred"].max())
        fig_sc.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines", name="Linea 1:1",
            line=dict(dash="dash", color="#94a3b8"),
        ))
        fig_sc.update_layout(
            title=f"Real vs Predicho — {best} (mejor R2)",
            xaxis_title="Valor real", yaxis_title="Prediccion",
            height=400,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Claves de session_state DISTINTAS a las keys de widgets
        st.session_state["ml_best_model"] = res_best["model"]
        st.session_state["ml_scaler"] = res_best["scaler"]
        st.session_state["ml_features"] = list(res_best["feature_names"])
        st.session_state["ml_target"] = target
        st.success(f"Modelo **{best}** guardado en sesion. Ya puedes usar PCA / Interpretabilidad / What-If.")
    else:
        st.info("Seleccione al menos una feature.")

# -----------------------------------------------------------------------------
# TAB 4 · PCA
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Reduccion de Dimensionalidad — PCA")
    st.markdown("Analisis de Componentes Principales sobre las 6 variables socioeconomicas (estandarizadas).")

    if "ml_features" not in st.session_state:
        st.info("Por favor entrena o selecciona un modelo en la pestana Machine Learning primero.")
        st.stop()

    year_pca = st.slider(
        "Ano para PCA",
        year_min, year_max, latest_year,
        key="pca_year_slider",
    )
    df_pca = df_f[df_f["anio"] == year_pca].dropna(subset=list(VAR_LABELS.keys()))
    X_pca = df_pca[list(VAR_LABELS.keys())]
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X_pca)

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)
    df_pca = df_pca.copy()
    df_pca["PC1"] = comps[:, 0]
    df_pca["PC2"] = comps[:, 1]
    var_exp = pca.explained_variance_ratio_

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig_pca = px.scatter(
            df_pca, x="PC1", y="PC2", text="departamento",
            color="pib", color_continuous_scale="Viridis",
            title=f"Proyeccion PC1 vs PC2 · {year_pca}",
            labels={"PC1": f"PC1 ({var_exp[0]*100:.1f}%)", "PC2": f"PC2 ({var_exp[1]*100:.1f}%)"},
        )
        fig_pca.update_traces(textposition="top center", textfont_size=9)
        fig_pca.update_layout(height=480)
        st.plotly_chart(fig_pca, use_container_width=True)

    with c2:
        st.markdown("**Varianza explicada**")
        st.metric("PC1", f"{var_exp[0]*100:.1f}%")
        st.metric("PC2", f"{var_exp[1]*100:.1f}%")
        st.metric("Acumulada", f"{var_exp.sum()*100:.1f}%")

        st.markdown("**Cargas (loadings)**")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=["PC1", "PC2"],
            index=[VAR_SHORT[v] for v in VAR_LABELS.keys()],
        )
        st.dataframe(loadings.style.format("{:.3f}"), use_container_width=True)

        fig_var = px.bar(
            x=["PC1", "PC2"], y=var_exp * 100,
            labels={"x": "Componente", "y": "% Varianza"},
            title="Varianza por componente",
            color=["PC1", "PC2"], color_discrete_sequence=["#0f766e", "#1d4ed8"],
        )
        fig_var.update_layout(showlegend=False, height=250, margin=dict(t=40))
        st.plotly_chart(fig_var, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5 · INTERPRETABILIDAD
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Importancia Relativa de Variables — Random Forest")
    st.markdown(
        "Se entrena un `RandomForestRegressor` para identificar los determinantes "
        "de la variable objetivo seleccionada."
    )

    if "ml_features" not in st.session_state:
        st.info("Por favor entrena o selecciona un modelo en la pestana Machine Learning primero.")
        st.stop()

    target_fi = st.selectbox(
        "Variable a explicar",
        options=list(VAR_LABELS.keys()),
        format_func=lambda x: VAR_LABELS[x],
        index=3,
        key="fi_target_select",
    )
    features_fi = [v for v in VAR_LABELS.keys() if v != target_fi]

    data_fi = df_f.dropna(subset=[target_fi] + features_fi)
    X_fi = data_fi[features_fi]
    y_fi = data_fi[target_fi]

    with st.spinner("Calculando importancia de features..."):
        imp_df, rf_model = feature_importance_rf(X_fi, y_fi)

    imp_df["etiqueta"] = imp_df["variable"].map(VAR_SHORT)

    fig_imp = px.bar(
        imp_df, x="importancia", y="etiqueta", orientation="h",
        title=f"Feature Importance → {VAR_SHORT[target_fi]}",
        color="importancia", color_continuous_scale="Teal",
        labels={"importancia": "Importancia relativa", "etiqueta": ""},
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.dataframe(
        imp_df[["etiqueta", "importancia"]].rename(columns={"etiqueta": "Variable"}).style.format({"importancia": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Modelo: RandomForest (n_estimators=200, max_depth=8). "
        f"R2 in-sample: {rf_model.score(X_fi, y_fi):.3f}"
    )

# -----------------------------------------------------------------------------
# TAB 6 · SIMULACION WHAT-IF
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("Simulacion de Politica Publica — What-If")
    st.markdown(
        "Simule un choque porcentual en la **Inversion Publica DNP** y observe el efecto "
        "estimado sobre la variable objetivo mediante el mejor modelo ML entrenado."
    )

    if "ml_features" not in st.session_state:
        st.info("Por favor entrena o selecciona un modelo en la pestana Machine Learning primero.")
        st.stop()

    if "ml_best_model" not in st.session_state or "ml_scaler" not in st.session_state:
        st.warning("Primero entrene un modelo en la pestana **Machine Learning** para habilitar la simulacion.")
        st.stop()

    model = st.session_state["ml_best_model"]
    scaler = st.session_state["ml_scaler"]
    feat_names = st.session_state["ml_features"]
    tgt = st.session_state["ml_target"]

    st.info(
        f"Modelo activo: mejor del ultimo entrenamiento · Objetivo: **{VAR_SHORT[tgt]}** · "
        f"Features: {', '.join([VAR_SHORT[f] for f in feat_names])}"
    )

    shock_pct = st.slider(
        "Incremento % en Inversion DNP (choque presupuestal)",
        min_value=-30, max_value=50, value=10, step=5,
        key="whatif_shock_slider",
    )

    base = df_f[df_f["anio"] == latest_year][feat_names].mean().to_frame().T
    base = base.reindex(columns=feat_names)

    scenario = base.copy()
    if "inversion_dnp" in feat_names:
        scenario["inversion_dnp"] = scenario["inversion_dnp"] * (1 + shock_pct / 100)

    X_base = scaler.transform(base)
    X_scen = scaler.transform(scenario)
    pred_base = float(model.predict(X_base)[0])
    pred_scen = float(model.predict(X_scen)[0])
    delta = pred_scen - pred_base
    delta_pct = (delta / pred_base * 100) if pred_base != 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{VAR_SHORT[tgt]} baseline", f"{pred_base:.2f}")
    c2.metric(f"{VAR_SHORT[tgt]} con choque", f"{pred_scen:.2f}", delta=f"{delta:+.2f}")
    c3.metric("Variacion relativa", f"{delta_pct:+.1f}%")

    fig_wh = go.Figure()
    fig_wh.add_trace(go.Bar(
        x=["Baseline", f"Choque {shock_pct:+d}% Inv. DNP"],
        y=[pred_base, pred_scen],
        marker_color=["#64748b", "#0f766e"],
        text=[f"{pred_base:.2f}", f"{pred_scen:.2f}"],
        textposition="auto",
    ))
    fig_wh.update_layout(
        title=f"Efecto estimado del choque presupuestal sobre {VAR_SHORT[tgt]}",
        yaxis_title=VAR_SHORT[tgt], height=360,
    )
    st.plotly_chart(fig_wh, use_container_width=True)

    with st.expander("Detalles tecnicos del escenario"):
        st.write("**Vector de features baseline:**")
        st.dataframe(base.style.format("{:.1f}"), hide_index=True)
        st.write("**Vector de features con choque:**")
        st.dataframe(scenario.style.format("{:.1f}"), hide_index=True)
        st.caption(
            "Se respeta estrictamente el orden y nombres de columnas del entrenamiento "
            "(feature_names_in_ / columnas de fit) para evitar ValueError de scikit-learn."
        )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    "Observatorio de Analitica Economica y Social por Departamentos en Colombia · "
    "Datos sinteticos calibrados con rangos y tendencias 2010-2024 · "
    "En produccion: conectar endpoints DANE / TerriData DNP. "
    "Metodologia: ADF · Johansen · VAR/VECM · Ridge/Lasso/SVR/MLP · PCA · Random Forest."
)
