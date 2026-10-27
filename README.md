# Observatorio de Analítica Económica y Social por Departamentos en Colombia

Plataforma Streamlit de nivel *Think Tank* para el análisis regional colombiano (2010–2024).

## Módulos

1. **EDA Espacial** — Mapas coropléticos interactivos (Plotly) + ranking y series temporales.
2. **Econometría de Series de Tiempo** — ADF, Johansen, selección automática VAR/VECM y pronóstico.
3. **Machine Learning Supervisado** — Ridge, Lasso, SVR, MLPRegressor + métricas R²/RMSE y scatter 1:1.
4. **PCA** — Reducción de dimensionalidad con varianza explicada y loadings.
5. **Interpretabilidad** — Feature Importance vía RandomForestRegressor.
6. **Simulación What-If** — Choques presupuestales en Inversión DNP con control estricto de nombres de features.

## Variables del panel

| Variable | Descripción |
|----------|-------------|
| `pib` | PIB departamental (miles de millones COP) |
| `homicidios` | Tasa de homicidios por 100.000 hab. |
| `inversion_dnp` | Inversión pública DNP (millones COP) |
| `ipm` | Índice de Pobreza Multidimensional (%) |
| `desempleo` | Tasa de desempleo anual (%) |
| `cobertura_edu` | Cobertura neta educación media (%) |

## Ejecución local

```bash
cd observatorio_colombia
pip install -r requirements.txt
streamlit run app.py
```

## Notas de datos

- El pipeline intenta cargar un CSV local o un endpoint HTTP.
- Si falla, carga automáticamente un **dataset maestro sintético** calibrado con rangos, tendencias y correlaciones realistas (incluyendo shock COVID-19).
- Gaps se imputan con interpolación lineal *intragrupo* por departamento.
- GeoJSON de departamentos se descarga desde un repositorio público (DANE MGN 2018).

## Arquitectura

- `@st.cache_data` en todas las cargas pesadas.
- Cero excepciones no capturadas en pantalla.
- CSS personalizado para KPI cards y tipografía profesional.
- Sidebar con filtros globales; 6 pestañas modulares.
