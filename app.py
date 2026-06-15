import base64
import csv
import io
import math
import os
import pickle
import time
from pathlib import Path 

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, redirect, render_template, request, send_file, url_for
from plotly.offline import plot

app = Flask(__name__)

STATIC_LOGO_PATH = Path("static/logo_up2.png")
FALLBACK_LOGO_PATH = Path(r"C:\Users\mi23346\Desktop\codex_test\static\logo_up2.png")

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    import umap
except ImportError:
    umap = None

try:
    import optuna
except ImportError:
    optuna = None

try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None

try:
    from AutoClean import AutoClean
except ImportError:
    AutoClean = None

try:
    from sklearn.inspection import partial_dependence
except ImportError:
    partial_dependence = None

try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    import lime.lime_tabular
except ImportError:
    lime = None

from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler


TEAM_MEMBERS = [
    {"name": "César Augusto Pérez Rosas", "email": "0287285@up.edu.mx"},
    {"name": "Luis Antonio Mani Yáñez", "email": "0287239@up.edu.mx"},
    {"name": "Oscar Barranco Velázquez", "email": "0287244@up.edu.mx"},
]
PROFESSOR_NAME = "Dr. León Palafox"

MODEL_NOTES = [
    {
        "name": "Regresión Lineal",
        "description": "Modelo base que estima relaciones lineales entre variables.",
        "benefits": "Rápido, estable y muy interpretable.",
        "strengths": "Buen baseline, bajo costo y coeficientes claros.",
        "weaknesses": "Pierde fuerza cuando la relación real no es lineal.",
        "bias_variance": "Sesgo relativamente alto y varianza baja.",
        "interpretability": "Muy alta.",
    },
    {
        "name": "Random Forest",
        "description": "Ensamble de muchos árboles entrenados con subconjuntos del dataset.",
        "benefits": "Robusto y con buen desempeño general.",
        "strengths": "Captura no linealidades y controla mejor el sobreajuste que un árbol único.",
        "weaknesses": "Más pesado y menos transparente que un modelo lineal.",
        "bias_variance": "Bias medio y varianza controlada por el ensamble.",
        "interpretability": "Media.",
    },
    {
        "name": "Gradient Boosting",
        "description": "Construye árboles secuenciales que corrigen errores previos.",
        "benefits": "Muy competitivo en datos tabulares.",
        "strengths": "Suele lograr gran precisión si está bien ajustado.",
        "weaknesses": "Más sensible a hiperparámetros.",
        "bias_variance": "Reduce bias, pero puede subir varianza si se forza demasiado.",
        "interpretability": "Media-baja.",
    },
    {
        "name": "XGBoost",
        "description": "Boosting optimizado con regularización y gran eficiencia.",
        "benefits": "Muy potente para datasets tabulares.",
        "strengths": "Flexibilidad y alto rendimiento.",
        "weaknesses": "Más complejo de configurar y explicar.",
        "bias_variance": "Muy flexible; requiere control para no sobreajustar.",
        "interpretability": "Media-baja.",
    }
]

APP_STATE = {
    "regression_dataset": None,
    "regression_models": [],
    "csv_state": None,
    "csv_predictions": None,
    "dimensionality": None,
    "tuned_model": None,
    "tuning_result": None,
    "profile_html": None,
}

XAI_VIEWS = {
    "regression": {"available": False, "chart": "", "message": "XAI no disponible."},
}


def get_logo_path():
    if STATIC_LOGO_PATH.exists():
        return STATIC_LOGO_PATH
    if FALLBACK_LOGO_PATH.exists():
        return FALLBACK_LOGO_PATH
    return None


def load_regression_dataset():
    dataset = load_diabetes()
    frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    frame["target"] = dataset.target
    return {
        "name": "Diabetes Dataset",
        "description": "Dataset de regresión con 442 registros y 10 variables para estimar progresión de diabetes.",
        "frame": frame,
        "feature_names": list(dataset.feature_names),
        "target_name": "target",
    }


def split_dataset(dataset):
    x = dataset["frame"][dataset["feature_names"]]
    y = dataset["frame"][dataset["target_name"]]
    return train_test_split(x, y, test_size=0.2, random_state=42)


def regression_builders():
    builders = {
        "linear_regression": ("Regresión Lineal", lambda params=None: LinearRegression()),
        "random_forest": (
            "Random Forest Regressor",
            lambda params=None: RandomForestRegressor(random_state=42, **(params or {
                "n_estimators": 220,
                "max_depth": 10,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
            })),
        ),
        "gradient_boosting": (
            "Gradient Boosting Regressor",
            lambda params=None: GradientBoostingRegressor(random_state=42, **(params or {
                "n_estimators": 220,
                "learning_rate": 0.04,
                "max_depth": 2,
                "subsample": 0.9,
            })),
        ),
    }
    if XGBRegressor is not None:
        builders["xgboost"] = (
            "XGBoost Regressor",
            lambda params=None: XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                **(params or {
                    "n_estimators": 260,
                    "max_depth": 3,
                    "learning_rate": 0.04,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                }),
            ),
        )
    return builders


def train_regression_models(dataset):
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    models = []
    for key, (label, builder) in regression_builders().items():
        model = builder()
        model.fit(x_train, y_train)
        preds_train = model.predict(x_train)
        preds_test = model.predict(x_test)
        mse = mean_squared_error(y_test, preds_test)
        
        try:
            raw_params = model.get_params()
            important_keys = ['n_estimators', 'max_depth', 'learning_rate', 'subsample']
            filtered = {k: v for k, v in raw_params.items() if k in important_keys and v is not None}
            if not filtered: 
                filtered = {k: v for k, v in raw_params.items() if isinstance(v, (int, float, str, bool))}
            params_str = ", ".join([f"{k}={v}" for k, v in filtered.items()])
        except Exception:
            params_str = "Por defecto"

        models.append({
            "key": key,
            "label": label,
            "model": model,
            "hyperparams_str": params_str,
            "metrics": {
                "r2_train": r2_score(y_train, preds_train),
                "r2": r2_score(y_test, preds_test),
                "mse": mse,
                "rmse": math.sqrt(mse),
                "mae": mean_absolute_error(y_test, preds_test),
            },
            "hyperparameters": model.get_params(),
        })
    return sorted(models, key=lambda item: item["metrics"]["r2"], reverse=True)


def plot_div(fig):
    return plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def metric_bar_chart(models, metric_key, title, color):
    fig = go.Figure([
        go.Bar(
            x=[m["label"] for m in models],
            y=[m["metrics"][metric_key] for m in models],
            text=[f'{m["metrics"][metric_key]:.4f}' for m in models],
            textposition="outside",
            cliponaxis=False,
            marker=dict(color=color),
        )
    ])
    fig.update_layout(
        title=title,
        paper_bgcolor="#fffaf2",
        plot_bgcolor="#fffaf2",
        height=430,
        margin=dict(l=50, r=20, t=80, b=120),
    )
    fig.update_xaxes(tickangle=-18, automargin=True)
    fig.update_yaxes(gridcolor="#ead7b1")
    return plot_div(fig)


def dataset_exploration(dataset):
    frame = dataset["frame"]
    features = dataset["feature_names"][:6]
    hist_fig = go.Figure()
    for column in features[:3]:
        hist_fig.add_trace(go.Histogram(x=frame[column], name=column, opacity=0.65))
    hist_fig.update_layout(barmode="overlay", title="Histogramas de variables", paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=420)

    box_fig = go.Figure()
    for column in features[:4]:
        box_fig.add_trace(go.Box(y=frame[column], name=column))
    box_fig.update_layout(title="Diagramas de caja", paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=420)

    corr = frame[dataset["feature_names"][:8]].corr()
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="YlOrRd",
        )
    )
    heatmap.update_layout(title="Correlaciones", paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=460)

    return {
        "hist": plot_div(hist_fig),
        "box": plot_div(box_fig),
        "corr": plot_div(heatmap),
    }


def build_profile_html(dataset):
    if ProfileReport is None:
        return None
    sample = dataset["frame"].copy()
    report = ProfileReport(sample, minimal=True, explorative=False, title=f"Perfil de {dataset['name']}")
    html = report.to_html()
    return base64.b64encode(html.encode("utf-8")).decode("utf-8")


def dimensionality_views(dataset, tsne_perplexity=30, umap_neighbors=15, umap_min_dist=0.1):
    x = dataset["frame"][dataset["feature_names"]]
    y = dataset["frame"][dataset["target_name"]]
    scaled = StandardScaler().fit_transform(x)[:220]
    labels = y.iloc[:220].tolist()

    pca_points = PCA(n_components=2, random_state=42).fit_transform(scaled)
    tsne_points = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42, perplexity=tsne_perplexity).fit_transform(scaled)

    def scatter(points, title):
        fig = go.Figure([go.Scattergl(
            x=points[:, 0], y=points[:, 1], mode="markers",
            marker=dict(size=9, color=labels, colorscale="YlOrRd", showscale=True, opacity=0.8, colorbar=dict(title="Target")),
        )])
        fig.update_layout(title=title, paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=420)
        fig.update_xaxes(gridcolor="#ead7b1")
        fig.update_yaxes(gridcolor="#ead7b1")
        return plot_div(fig)

    umap_chart = ""
    umap_available = False
    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=umap_neighbors, min_dist=umap_min_dist)
        umap_points = reducer.fit_transform(scaled)
        umap_chart = scatter(umap_points, "UMAP del dataset de regresión")
        umap_available = True

    return {
        "pca_chart": scatter(pca_points, "PCA del dataset de regresión"),
        "tsne_chart": scatter(tsne_points, "t-SNE del dataset de regresión"),
        "umap_chart": umap_chart,
        "umap_available": umap_available,
        "params": {
            "tsne_perplexity": tsne_perplexity,
            "umap_neighbors": umap_neighbors,
            "umap_min_dist": umap_min_dist,
        },
    }


def xai_feature_name(dataset):
    frame = dataset["frame"]
    target = dataset["target_name"]
    ranked = []
    for feature in dataset["feature_names"]:
        series = frame[feature]
        if series.nunique() < 2:
            continue
        corr = series.corr(frame[target])
        if pd.isna(corr):
            continue
        ranked.append((abs(float(corr)), feature))
    if not ranked:
        return dataset["feature_names"][0]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def xai_pdp_ice_view(dataset, model_data, feature_name):
    if partial_dependence is None or model_data is None:
        return {"available": False, "chart": "", "message": "No se detectó sklearn.inspection.partial_dependence."}

    x_all = dataset["frame"][dataset["feature_names"]]
    x_for_pd = x_all.sample(n=min(len(x_all), 220), random_state=42)
    x_for_pd.columns = [str(col) for col in x_for_pd.columns]
    feature_name_clean = str(feature_name)
    
    try:
        pd_result = partial_dependence(
            estimator=model_data["model"], 
            X=x_for_pd, 
            features=[feature_name_clean],
            kind="both", 
            grid_resolution=40
        )
        grid_values = pd_result["grid_values"] if "grid_values" in pd_result else pd_result["values"]
        x_axis = list(grid_values[0])
        average = pd_result["average"][0]
        individual_raw = pd_result["individual"]
        individual = individual_raw[0] if len(individual_raw.shape) == 3 else individual_raw
        
        total_lines = int(individual.shape[0]) if hasattr(individual, "shape") else len(individual)
        max_lines = min(50, total_lines)
        if total_lines > max_lines and max_lines > 0:
            step = max(1, total_lines // max_lines)
            selected_indexes = list(range(0, total_lines, step))[:max_lines]
        else:
            selected_indexes = list(range(max_lines))

        fig = go.Figure()
        ice_color = "rgba(139, 30, 42, 0.15)"
        for idx in selected_indexes:
            row = individual[idx]
            fig.add_trace(go.Scatter(
                x=x_axis, y=row.tolist() if hasattr(row, "tolist") else list(row),
                mode="lines", line=dict(color=ice_color, width=1), opacity=0.25, hoverinfo="skip", showlegend=False
            ))
            
        pdp_color = "#8b1e2a"
        fig.add_trace(go.Scatter(
            x=x_axis, y=average.tolist() if hasattr(average, "tolist") else list(average),
            mode="lines", line=dict(color=pdp_color, width=4.5), name="PDP (promedio)"
        ))
        
        fig.update_layout(
            title=f"Impacto de {feature_name} usando {model_data['label']}",
            paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=380, margin=dict(l=45, r=15, t=50, b=45)
        )
        fig.update_xaxes(gridcolor="#ead7b1")
        fig.update_yaxes(gridcolor="#ead7b1")

        return {
            "available": True,
            "chart": plot_div(fig),
            "message": f"ICE mostrado con {len(selected_indexes)} trayectorias (máximo 50). Variable analizada: {feature_name}",
        }

    except Exception as e:
        return {"available": False, "chart": "", "message": f"Error al calcular PDP/ICE: {str(e)}"}
    
    
def generate_shap_plot(model_data, dataset, feature_name, shap_type="summary_bar"):
    if shap is None or model_data is None:
        return "<p class='subtle'>Librería SHAP no disponible en el entorno.</p>"
    
    import numpy as np
    X = dataset["frame"][dataset["feature_names"]]
    X_sample = X.sample(min(100, len(X)), random_state=42)
    
    try:
        model = model_data["model"]
        model_key = model_data["key"]
        
        if any(k in model_key for k in ["forest", "boosting", "xgboost"]):
            explainer = shap.TreeExplainer(model)
            try:
                shap_values_raw = explainer.shap_values(X_sample, check_additivity=False)
            except Exception:
                shap_values_raw = explainer.shap_values(X_sample)
        else:
            try:
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values_raw = explainer.shap_values(X_sample)
            except Exception:
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values_raw = explainer(X_sample).values

        if isinstance(shap_values_raw, list):
            vals = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
        elif hasattr(shap_values_raw, "values"):
            vals = shap_values_raw.values
            if len(vals.shape) == 3:
                vals = vals[:, :, 1]
        elif len(shap_values_raw.shape) == 3:
            vals = shap_values_raw[:, :, 1]
        else:
            vals = shap_values_raw

        app_solid_color = "#8b1e2a"
        shap_colorscale = [[0.0, '#1E88E5'], [0.5, '#D3D3D3'], [1.0, '#FF0052']]

        if shap_type == "summary_bar":
            mean_abs_shap = np.abs(vals).mean(axis=0)
            df_shap = pd.DataFrame({"feature": dataset["feature_names"], "importance": mean_abs_shap})
            df_shap = df_shap.sort_values(by="importance", ascending=True).tail(8)
            
            fig = go.Figure(go.Bar(
                x=df_shap["importance"], y=df_shap["feature"],
                orientation="h", marker=dict(color=app_solid_color)
            ))
            fig.update_layout(
                title=f"Importancia Global SHAP ({model_data['label']})",
                paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=320,
                margin=dict(l=140, r=20, t=40, b=40)
            )
            fig.update_xaxes(gridcolor="#ead7b1", title="|SHAP Value| promedio", exponentformat="none")
            fig.update_yaxes(gridcolor="#ead7b1")
            return plot_div(fig)
            
        elif shap_type == "summary_beeswarm":
            mean_abs_shap = np.abs(vals).mean(axis=0)
            indices_ordenados = np.argsort(mean_abs_shap)
            top_indices = indices_ordenados[-8:] 
            
            x_coords = []
            y_coords = []
            color_vals = []
            
            for y_idx, feat_idx in enumerate(top_indices):
                feature_name_curr = dataset["feature_names"][feat_idx]
                feat_shap = vals[:, feat_idx]
                feat_real = X_sample[feature_name_curr].values
                
                f_min, f_max = feat_real.min(), feat_real.max()
                feat_norm = (feat_real - f_min) / (f_max - f_min) if f_max > f_min else np.zeros_like(feat_real)
                
                np.random.seed(42) 
                jitter = np.random.uniform(-0.18, 0.18, size=len(feat_shap))
                
                for i in range(len(feat_shap)):
                    x_coords.append(feat_shap[i])
                    y_coords.append(y_idx + jitter[i])
                    color_vals.append(feat_norm[i])
                    
            fig = go.Figure(go.Scatter(
                x=x_coords, y=y_coords, mode="markers",
                marker=dict(
                    color=color_vals, colorscale=shap_colorscale, showscale=True,
                    colorbar=dict(title="Valor Real", tickvals=[0, 1], ticktext=["Bajo", "Alto"], thickness=12, len=0.85),
                    size=7, opacity=0.85, line=dict(width=0.4, color="#2e2418")
                )
            ))
            
            top_names = [dataset["feature_names"][idx] for idx in top_indices]
            fig.update_layout(
                title=f"Distribución de Impactos SHAP Beeswarm ({model_data['label']})",
                paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=340,
                margin=dict(l=150, r=20, t=40, b=40)
            )
            fig.update_yaxes(tickmode="array", tickvals=list(range(len(top_names))), ticktext=top_names, gridcolor="#ead7b1")
            fig.update_xaxes(gridcolor="#ead7b1", title="SHAP Value (Impacto)", exponentformat="none",
                             zeroline=True, zerolinecolor="#2e2418", zerolinewidth=1.5)
            return plot_div(fig)
            
        else:
            feat_idx = dataset["feature_names"].index(feature_name)
            x_vals = X_sample[feature_name].values
            y_vals = vals[:, feat_idx]
            
            fig = go.Figure(go.Scatter(
                x=x_vals, y=y_vals, mode="markers",
                marker=dict(
                    color=x_vals, colorscale=shap_colorscale, showscale=True,
                    colorbar=dict(title="Valor Real", thickness=15, len=0.85),
                    size=10, opacity=0.9, line=dict(width=1.2, color="#2e2418")
                )
            ))
            fig.update_layout(
                title=f"Dependencia SHAP: {feature_name}",
                paper_bgcolor="#fffaf2", plot_bgcolor="#fffaf2", height=320,
                margin=dict(l=65, r=20, t=40, b=40)
            )
            fig.update_xaxes(gridcolor="#ead7b1", title=f"Valor real de {feature_name}", exponentformat="none")
            fig.update_yaxes(gridcolor="#ead7b1", title="Valor SHAP (Impacto)", exponentformat="none",
                             zeroline=True, zerolinecolor="#2e2418", zerolinewidth=2)
            return plot_div(fig)
        
    except Exception as e:
        return f"<p class='subtle'>Gráfico SHAP no disponible para este algoritmo: {str(e)}</p>"


def generate_lime_explanation(model_data, dataset):
    if lime is None or model_data is None:
        return "<p class='subtle'>Librería LIME no instalada en el servidor.</p>"
    try:
        X = dataset["frame"][dataset["feature_names"]]
        X_sample = X.iloc[0] 
        
        mode = "regression"
        predict_fn = model_data["model"].predict
        class_names = [dataset["target_name"]]
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=dataset["feature_names"],
            class_names=class_names,
            mode=mode,
            random_state=42
        )
        
        exp = explainer.explain_instance(
            data_row=X_sample.values,
            predict_fn=predict_fn,
            num_features=6
        )
        
        # --- EXTRACCIÓN MATEMÁTICA PARA PLOTLY ---
        exp_list = exp.as_list()
        exp_list.reverse() # Invertir para que las barras de mayor impacto queden arriba
        
        features = [item[0] for item in exp_list]
        weights = [item[1] for item in exp_list]
        # Usamos los colores de tu app: Rojo si empuja la predicción arriba, Marino si la baja
        colors = ['#8b1e2a' if w > 0 else '#17324d' for w in weights] 

        fig = go.Figure(go.Bar(
            x=weights,
            y=features,
            orientation='h',
            marker_color=colors
        ))

        local_pred = exp.local_pred[0]
        
        fig.update_layout(
            title=f"LIME Explicación Local (Predicción del modelo: {local_pred:.2f})",
            paper_bgcolor="#fffaf2",
            plot_bgcolor="#fffaf2",
            height=340,
            margin=dict(l=160, r=20, t=50, b=40)
        )
        fig.update_xaxes(gridcolor="#ead7b1", title="Impacto en la predicción (Pesos LIME)", zeroline=True, zerolinecolor="#2e2418", zerolinewidth=1.5)
        fig.update_yaxes(gridcolor="#ead7b1")

        return plot_div(fig)
        
    except Exception as e:
        return f"<p class='subtle'>No se pudo generar explicación LIME: {str(e)}</p>"


def build_xai_views(regression_dataset, regression_models, 
                    reg_model_key=None, reg_feat=None, reg_shap_type="summary_bar"):
    
    reg_m = next((m for m in regression_models if m["key"] == reg_model_key), regression_models[0])
    r_feat = reg_feat if reg_feat else xai_feature_name(regression_dataset)

    return {
        "selected_reg_model": reg_m["key"], "selected_reg_feat": r_feat, "selected_reg_shap_type": reg_shap_type,
        "regression": xai_pdp_ice_view(regression_dataset, reg_m, r_feat),
        "reg_shap": generate_shap_plot(reg_m, regression_dataset, r_feat, reg_shap_type),
        "reg_lime": generate_lime_explanation(reg_m, regression_dataset)
    }


def predict_from_csv(csv_state):
    dataset = APP_STATE["regression_dataset"] 
    models = APP_STATE["regression_models"] 
    required = dataset["feature_names"]
    missing = [col for col in required if col not in csv_state["headers"]]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))
    df = pd.DataFrame(csv_state["rows"])
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        raise ValueError("No se encontraron filas válidas para generar predicciones.")

    tables = []
    for item in models:
        preds = item["model"].predict(df[required])
        rows = []
        for idx, pred in enumerate(preds[:15], start=1):
            value = round(float(pred), 4)
            rows.append({"row": idx, "prediction": value})
        tables.append({"model": item["label"], "predictions": rows})
    return {"record_count": len(df), "tables": tables, "required_features": required}


def parse_csv_upload(file_storage):
    text = file_storage.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader]
    return {
        "filename": file_storage.filename or "dataset.csv",
        "headers": reader.fieldnames or [],
        "rows": rows,
        "preview": rows[:8],
    }


def autclean_assessment():
    if AutoClean is None:
        return "AutoClean no está instalado. Para este proyecto no es indispensable; conviene más mantener limpieza explícita y controlada."
    return "AutoClean puede ayudar en pruebas rápidas de limpieza, pero para un proyecto académico conviene usarlo con cuidado porque automatiza decisiones que luego hay que explicar."


def parse_param_grid(raw_text):
    grid = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, values = line.split(":", 1)
        parsed = []
        for item in values.split(","):
            value = item.strip()
            if not value:
                continue
            if "." in value:
                parsed.append(float(value))
            else:
                parsed.append(int(value))
        if parsed:
            grid[key.strip()] = parsed
    return grid


def tune_model_online(model_key, raw_grid):
    grid = parse_param_grid(raw_grid)
    if not grid:
        raise ValueError("No se detectó una grilla válida de hiperparámetros.")

    dataset = APP_STATE["regression_dataset"]
    builder_map = regression_builders()
    score_key = "r2"

    if model_key not in builder_map:
        raise ValueError("Modelo no soportado para tuning.")

    x_train, x_test, y_train, y_test = split_dataset(dataset)
    label, builder = builder_map[model_key]

    t0 = time.time()
    base_model = builder()
    base_model.fit(x_train, y_train)
    base_preds = base_model.predict(x_test)
    base_score = r2_score(y_test, base_preds) 
    base_time = time.time() - t0

    t0 = time.time()
    best_gs_score = -float('inf')
    best_gs_model = None
    for params in ParameterGrid(grid):
        model = builder(params)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        score = r2_score(y_test, preds) 
        if score > best_gs_score:
            best_gs_score = score
            best_gs_model = model
    gs_time = time.time() - t0

    best_opt_score = -float('inf')
    best_opt_params = None
    best_opt_model = None
    opt_time = 0.0

    if optuna is not None:
        t0 = time.time()
        def objective(trial):
            trial_params = {}
            for k, v_list in grid.items():
                trial_params[k] = trial.suggest_categorical(k, v_list)
            
            model = builder(trial_params)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            return r2_score(y_test, preds) 
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        
        max_combos = 1
        for v in grid.values():
            max_combos *= len(v)
        n_trials = min(10, max_combos)
        
        study.optimize(objective, n_trials=n_trials)
        opt_time = time.time() - t0
        
        best_opt_score = study.best_value
        best_opt_params = study.best_params
        best_opt_model = builder(best_opt_params)
        best_opt_model.fit(x_train, y_train)
    else:
        best_opt_score = best_gs_score
        best_opt_params = best_gs_model.get_params()
        best_opt_model = best_gs_model

    comparison = [
        {"estrategia": "Modelo Base (Por Defecto)", "tiempo": f"{base_time:.4f} s", "score": base_score},
        {"estrategia": "Grid Search (Fuerza Bruta)", "tiempo": f"{gs_time:.4f} s", "score": best_gs_score},
    ]
    if optuna is not None:
        comparison.append({"estrategia": "Optuna (Muestreo Bayesiano)", "tiempo": f"{opt_time:.4f} s", "score": best_opt_score})

    APP_STATE["tuned_model"] = {"problem_type": "regression", "label": f"{label} ajustado", "model": best_opt_model}
    APP_STATE["tuning_result"] = {
        "problem_type": "regression",
        "label": label,
        "best_score": round(best_opt_score, 4),
        "score_key": score_key.upper(),
        "best_params": best_opt_params,
        "comparison": comparison,
    }


def initialize_state():
    global XAI_VIEWS
    APP_STATE["regression_dataset"] = load_regression_dataset()
    APP_STATE["regression_models"] = train_regression_models(APP_STATE["regression_dataset"])
    APP_STATE["dimensionality"] = dimensionality_views(APP_STATE["regression_dataset"])
    APP_STATE["profile_html"] = build_profile_html(APP_STATE["regression_dataset"])
    
    reg_m = APP_STATE["regression_models"][0]
    reg_f = xai_feature_name(APP_STATE["regression_dataset"])
    
    XAI_VIEWS = build_xai_views(
        APP_STATE["regression_dataset"], APP_STATE["regression_models"],
        reg_model_key=reg_m["key"], reg_feat=reg_f
    )


initialize_state()


@app.route("/up-logo")
def up_logo():
    logo_path = get_logo_path()
    if logo_path and logo_path.exists():
        return send_file(logo_path)
    return ("Logo no encontrado", 404)


@app.route("/download-tuned-model")
def download_tuned_model():
    tuned = APP_STATE.get("tuned_model")
    if not tuned:
        return redirect(url_for("home", tab="modelos"))
    buffer = io.BytesIO()
    pickle.dump(tuned["model"], buffer)
    buffer.seek(0)
    filename = tuned["label"].replace(" ", "_").lower() + ".pkl"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/octet-stream")


@app.route("/", methods=["GET", "POST"])
def home():
    active_tab = request.args.get("tab", request.form.get("tab", "resumen"))
    message = ""
    csv_message = ""
    prediction_rows = []
    csv_predictions = APP_STATE.get("csv_predictions")

    if request.method == "POST" and request.form.get("action") == "upload_csv":
        active_tab = "csv"
        csv_file = request.files.get("csv_file")
        if csv_file and csv_file.filename:
            APP_STATE["csv_state"] = parse_csv_upload(csv_file)
            APP_STATE["csv_predictions"] = None
            csv_message = "CSV cargado correctamente."
        else:
            csv_message = "Selecciona un archivo CSV."

    elif request.method == "POST" and request.form.get("action") == "predict_csv":
        active_tab = "csv"
        csv_state = APP_STATE.get("csv_state")
        if not csv_state:
            csv_message = "Primero carga un CSV."
        else:
            try:
                APP_STATE["csv_predictions"] = predict_from_csv(csv_state)
                csv_predictions = APP_STATE["csv_predictions"]
                csv_message = f"Se detectaron {csv_predictions['record_count']} registros y estas son las predicciones para cada modelo."
            except ValueError as error:
                csv_message = str(error)

    elif request.method == "POST" and request.form.get("action") == "predict_manual":
        active_tab = "prediccion"
        feature_names = APP_STATE["regression_dataset"]["feature_names"]
        try:
            raw_values = [float(request.form.get(feature, "0")) for feature in feature_names]
            structured_row = pd.DataFrame([raw_values], columns=feature_names)
            
            for model_data in APP_STATE["regression_models"]:
                value = float(model_data["model"].predict(structured_row)[0])
                prediction_rows.append({"label": model_data["label"], "prediction": round(value, 4)})
            message = "Predicciones generadas correctamente."
        except ValueError:
            message = "Introduce valores numéricos válidos."

    elif request.method == "POST" and request.form.get("action") == "update_embeddings":
        active_tab = "reduccion"
        try:
            APP_STATE["dimensionality"] = dimensionality_views(
                APP_STATE["regression_dataset"],
                tsne_perplexity=max(5.0, min(float(request.form.get("tsne_perplexity", "30")), 60.0)),
                umap_neighbors=max(2, min(int(request.form.get("umap_neighbors", "15")), 80)),
                umap_min_dist=max(0.0, min(float(request.form.get("umap_min_dist", "0.1")), 0.99)),
            )
            message = "Clusters actualizados."
        except ValueError:
            message = "No se pudieron actualizar los hiperparámetros."

    elif request.method == "POST" and request.form.get("action") == "tune_model":
        active_tab = "modelos"
        try:
            tune_model_online(
                request.form.get("model_key", ""),
                request.form.get("param_grid", ""),
            )
            message = "Reentrenamiento y evaluación comparativa completados."
        except ValueError as error:
            message = str(error)
        
    elif request.method == "POST" and request.form.get("action") == "update_xai":
        active_tab = "xai"
        global XAI_VIEWS
        
        reg_model = request.form.get("reg_model_key") or XAI_VIEWS.get("selected_reg_model")
        reg_feat = request.form.get("reg_feature") or XAI_VIEWS.get("selected_reg_feat")
        reg_shap = request.form.get("reg_shap_type") or XAI_VIEWS.get("selected_reg_shap_type", "summary_bar")
        if reg_shap == "summary": reg_shap = "summary_bar"
        
        XAI_VIEWS = build_xai_views(
            APP_STATE["regression_dataset"], APP_STATE["regression_models"],
            reg_model_key=reg_model, reg_feat=reg_feat, reg_shap_type=reg_shap
        )
        message = "Parámetros de explicabilidad actualizados correctamente."
        
    regression_models = APP_STATE["regression_models"]
    tuning_result = APP_STATE.get("tuning_result")

    return render_template(
        "index.html",
        active_tab=active_tab,
        message=message,
        csv_message=csv_message,
        up_logo_exists=get_logo_path() is not None,
        professor_name=PROFESSOR_NAME,
        team_members=TEAM_MEMBERS,
        regression_dataset=APP_STATE["regression_dataset"],
        regression_models=regression_models,
        regression_chart=metric_bar_chart(regression_models, "r2", "Regresión: comparación de R2", "#8b1e2a"),
        prediction_rows=prediction_rows,
        csv_state=APP_STATE["csv_state"],
        csv_predictions=csv_predictions,
        dimensionality=APP_STATE["dimensionality"],
        xai_views=XAI_VIEWS,
        model_notes=MODEL_NOTES,
        dataset_regression_plots=dataset_exploration(APP_STATE["regression_dataset"]),
        profile_html=APP_STATE["profile_html"],
        profile_available=ProfileReport is not None,
        autoclean_note=autclean_assessment(),
        tuning_result=tuning_result,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)