import csv
import io
import math
import os
from pathlib import Path

from flask import Flask, render_template, request, send_file

app = Flask(__name__)
UP_LOGO_PATH = Path(r"C:\Users\mi23346\Downloads\logo_UP.jpg")

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    import umap
except ImportError:
    umap = None

try:
    import optuna
except ImportError:
    optuna = None

import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REGRESSION_BUILDERS = {
    "linear_regression": ("Regresión Lineal", lambda: LinearRegression()),
    "random_forest": (
        "Random Forest",
        lambda: RandomForestRegressor(
            n_estimators=220,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
        ),
    ),
    "gradient_boosting": (
        "Gradient Boosting",
        lambda: GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=2,
            subsample=0.9,
            random_state=42,
        ),
    ),
}

if XGBRegressor is not None:
    REGRESSION_BUILDERS["xgboost"] = (
        "XGBoost",
        lambda: XGBRegressor(
            n_estimators=260,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.2,
            objective="reg:squarederror",
            random_state=42,
        ),
    )


CLASSIFICATION_BUILDERS = {
    "logistic_regression": (
        "Logistic Regression",
        lambda: LogisticRegression(max_iter=3000, solver="lbfgs"),
    ),
    "random_forest_classifier": (
        "Random Forest",
        lambda: RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
        ),
    ),
    "gradient_boosting_classifier": (
        "Gradient Boosting",
        lambda: GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    ),
}

if XGBClassifier is not None:
    CLASSIFICATION_BUILDERS["xgboost_classifier"] = (
        "XGBoost",
        lambda: XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
    )


HYPERPARAMETER_HELP = {
    "n_estimators": {
        "title": "Número de árboles",
        "description": "Cuántos árboles construye el ensamble. Más árboles suelen dar mayor estabilidad, pero suben costo y tiempo.",
        "icon": "forest",
    },
    "max_depth": {
        "title": "Profundidad máxima",
        "description": "Hasta dónde se puede ramificar cada árbol. Mucha profundidad detecta patrones complejos, pero puede sobreajustar.",
        "icon": "branches",
    },
    "learning_rate": {
        "title": "Velocidad de aprendizaje",
        "description": "Tamaño del ajuste en cada iteración de boosting. Más chico suele ser más estable, pero tarda más en aprender.",
        "icon": "speed",
    },
    "subsample": {
        "title": "Muestreo de filas",
        "description": "Porcentaje del dataset usado en cada ronda. Introduce diversidad y reduce sobreajuste.",
        "icon": "layers",
    },
    "colsample_bytree": {
        "title": "Muestreo de columnas",
        "description": "Porcentaje de variables candidatas en cada árbol. Ayuda a que el ensamble sea menos correlacionado.",
        "icon": "columns",
    },
    "min_samples_split": {
        "title": "Mínimo para dividir",
        "description": "Mínimo de observaciones para abrir una nueva rama. Evita divisiones muy frágiles.",
        "icon": "split",
    },
    "min_samples_leaf": {
        "title": "Mínimo por hoja",
        "description": "Mínimo de observaciones al final de una hoja. Suaviza decisiones demasiado finas.",
        "icon": "leaf",
    },
    "reg_alpha": {
        "title": "Regularización L1",
        "description": "Empuja a cero pesos poco útiles. Sirve para simplificar y evitar ruido.",
        "icon": "shield",
    },
    "reg_lambda": {
        "title": "Regularización L2",
        "description": "Reduce pesos exagerados y mejora estabilidad frente a outliers o ruido.",
        "icon": "shield",
    },
    "solver": {
        "title": "Optimizador",
        "description": "Método numérico usado para encontrar coeficientes en la regresión logística.",
        "icon": "gear",
    },
}

APP_STATE = {}
TEAM_MEMBERS = [
    {"name": "Jhosse Paul Márquez Ruiz", "email": "0287289@up.edu.mx"},
    {"name": "Cesar Augusto Pérez Rosas", "email": "0287285@up.edu.mx"},
    {"name": "Gerardo David Rivero Rique", "email": "0287274@up.edu.mx"},
    {"name": "Luis Antonio Mani Yáñez", "email": "0287239@up.edu.mx"},
    {"name": "Oscar Barranco Velázquez", "email": "0287244@up.edu.mx"},
]
PROFESSOR_NAME = "Dr. León Palafox"
MODEL_NOTES = [
    {
        "name": "Regresión Lineal",
        "description": "Modelo base que estima una relación lineal entre variables de entrada y la variable objetivo.",
        "benefits": "Rápido, simple, estable y muy fácil de explicar.",
        "strengths": "Alta interpretabilidad, bajo costo computacional, buen punto de referencia.",
        "weaknesses": "Puede subajustar cuando la relación real no es lineal.",
        "bias_variance": "Suele tener sesgo más alto y varianza baja.",
        "interpretability": "Muy alta: los coeficientes ayudan a entender el efecto de cada variable.",
    },
    {
        "name": "Random Forest",
        "description": "Ensamble de múltiples árboles entrenados sobre subconjuntos de datos y variables.",
        "benefits": "Robusto, flexible y con buen desempeño general.",
        "strengths": "Captura relaciones no lineales, reduce sobreajuste respecto a un árbol individual.",
        "weaknesses": "Menos interpretable y más pesado que un modelo lineal.",
        "bias_variance": "Bias medio y varianza controlada por el ensamble.",
        "interpretability": "Media: se puede analizar importancia de variables, pero no es tan transparente.",
    },
    {
        "name": "Gradient Boosting",
        "description": "Construye árboles secuenciales donde cada nuevo árbol corrige errores del anterior.",
        "benefits": "Muy competitivo en precisión y útil en patrones complejos.",
        "strengths": "Buen equilibrio entre flexibilidad y capacidad predictiva.",
        "weaknesses": "Más sensible a hiperparámetros y más lento de entrenar.",
        "bias_variance": "Reduce bias, pero si se configura mal puede aumentar varianza.",
        "interpretability": "Media-baja: mejor que una caja negra total, pero menos clara que un modelo lineal.",
    },
    {
        "name": "XGBoost",
        "description": "Versión optimizada de boosting con regularización y gran eficiencia.",
        "benefits": "Suele lograr alto desempeño y permite ajustes finos.",
        "strengths": "Muy potente, escalable y fuerte en datasets tabulares.",
        "weaknesses": "Mayor complejidad técnica y menor facilidad de explicación.",
        "bias_variance": "Puede reducir bias con fuerza, pero requiere control para no sobreajustar.",
        "interpretability": "Media-baja: se apoya en importancia de variables y análisis posterior.",
    },
]


def load_regression_dataset():
    dataset = load_diabetes()
    feature_names = list(dataset.feature_names)
    rows = []
    for index, target in enumerate(dataset.target):
        row = {feature: float(dataset.data[index][pos]) for pos, feature in enumerate(feature_names)}
        row["target"] = float(target)
        rows.append(row)
    return {
        "name": "Diabetes Dataset",
        "description": "Dataset de regresión con 442 registros y 10 variables para estimar progresión de diabetes.",
        "feature_names": feature_names,
        "rows": rows,
    }


def load_classification_dataset():
    dataset = load_breast_cancer()
    feature_names = list(dataset.feature_names)
    rows = []
    for index, target in enumerate(dataset.target):
        row = {feature: float(dataset.data[index][pos]) for pos, feature in enumerate(feature_names)}
        row["target"] = int(target)
        rows.append(row)
    return {
        "name": "Breast Cancer Wisconsin",
        "description": "Dataset categórico binario con 569 registros y 30 variables para clasificar tumores como benignos o malignos.",
        "feature_names": feature_names,
        "target_names": list(dataset.target_names),
        "rows": rows,
    }


def split_dataset(dataset):
    features = []
    targets = []
    for row in dataset["rows"]:
        features.append([row[feature] for feature in dataset["feature_names"]])
        targets.append(row["target"])
    return train_test_split(features, targets, test_size=0.2, random_state=42)


def train_regression_models(dataset):
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    models = []
    for model_key, (label, builder) in REGRESSION_BUILDERS.items():
        model = builder()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        models.append(
            {
                "key": model_key,
                "label": label,
                "model": model,
                "metrics": {
                    "r2": r2_score(y_test, predictions),
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                    "mae": mean_absolute_error(y_test, predictions),
                },
                "hyperparameters": model.get_params(),
            }
        )
    models.sort(key=lambda item: item["metrics"]["r2"], reverse=True)
    return models


def train_classification_models(dataset):
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    models = []
    for model_key, (label, builder) in CLASSIFICATION_BUILDERS.items():
        model = builder()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        models.append(
            {
                "key": model_key,
                "label": label,
                "model": model,
                "metrics": {
                    "accuracy": accuracy_score(y_test, predictions),
                    "f1": f1_score(y_test, predictions),
                    "precision": precision_score(y_test, predictions),
                    "recall": recall_score(y_test, predictions),
                },
                "hyperparameters": model.get_params(),
            }
        )
    models.sort(key=lambda item: item["metrics"]["accuracy"], reverse=True)
    return models


def make_plotly_bar(models, metric_key, title, color):
    labels = [model["label"] for model in models]
    values = [model["metrics"][metric_key] for model in models]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=color, line=dict(color="#8e6a22", width=1.2)),
                text=[f"{value:.4f}" for value in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Valor: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="#fffaf2",
        plot_bgcolor="#fffaf2",
        font=dict(color="#2e2418", size=15),
        margin=dict(l=50, r=25, t=90, b=120),
        height=470,
        bargap=0.22,
    )
    fig.update_xaxes(tickangle=-18, automargin=True, showgrid=False, tickfont=dict(size=13))
    fig.update_yaxes(showgrid=True, gridcolor="#ead7b1", automargin=True)
    return plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def make_plotly_prediction(prediction_rows):
    if not prediction_rows:
        return ""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[row["label"] for row in prediction_rows],
            y=[row["prediction"] for row in prediction_rows],
            mode="lines+markers+text",
            text=[f'{row["prediction"]:.2f}' for row in prediction_rows],
            textposition="top center",
            marker=dict(size=14, color=["#8b1e2a", "#b69149", "#0f766e", "#17324d"][: len(prediction_rows)]),
            line=dict(color="#8b1e2a", width=4),
            hovertemplate="<b>%{x}</b><br>Predicción: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Comparación visual de predicciones",
        paper_bgcolor="#fffaf2",
        plot_bgcolor="#fffaf2",
        font=dict(color="#2e2418", size=15),
        margin=dict(l=50, r=25, t=80, b=90),
        height=440,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#ead7b1", automargin=True)
    return plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def make_plotly_scatter(points, labels, title):
    colors = ["#b69149" if int(label) == 0 else "#8b1e2a" for label in labels]
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=[point[0] for point in points],
                y=[point[1] for point in points],
                mode="markers",
                marker=dict(size=9, color=colors, opacity=0.78),
                text=[f"Clase {label}" for label in labels],
                hovertemplate="%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="#fffaf2",
        plot_bgcolor="#fffaf2",
        font=dict(color="#2e2418", size=14),
        margin=dict(l=45, r=20, t=80, b=45),
        height=420,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ead7b1", automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#ead7b1", automargin=True)
    return plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def prepare_dimensionality_views(dataset, tsne_perplexity=30, umap_neighbors=15, umap_min_dist=0.1):
    x = [[row[feature] for feature in dataset["feature_names"]] for row in dataset["rows"]]
    y = [row["target"] for row in dataset["rows"]]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    capped_x = x_scaled[:220]
    capped_y = y[:220]

    pca_points = PCA(n_components=2, random_state=42).fit_transform(capped_x)
    tsne_points = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=42,
        perplexity=tsne_perplexity,
    ).fit_transform(capped_x)

    umap_plot = ""
    umap_available = False
    if umap is not None:
        umap_points = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
        ).fit_transform(capped_x)
        umap_plot = make_plotly_scatter(umap_points, capped_y, "UMAP del dataset categórico")
        umap_available = True

    return {
        "pca_chart": make_plotly_scatter(pca_points, capped_y, "PCA del dataset categórico"),
        "tsne_chart": make_plotly_scatter(tsne_points, capped_y, "t-SNE del dataset categórico"),
            "umap_chart": umap_plot,
            "umap_available": umap_available,
            "params": {
                "tsne_perplexity": tsne_perplexity,
                "umap_neighbors": umap_neighbors,
                "umap_min_dist": umap_min_dist,
            },
    }


def build_hyperparameter_cards(models):
    cards = []
    seen = set()
    for model in models:
        for key, value in model["hyperparameters"].items():
            if key in HYPERPARAMETER_HELP and key not in seen:
                seen.add(key)
                help_item = HYPERPARAMETER_HELP[key]
                cards.append(
                    {
                        "key": key,
                        "title": help_item["title"],
                        "description": help_item["description"],
                        "icon": help_item["icon"],
                        "example_value": value,
                    }
                )
    return cards


def build_conclusions(regression_models, classification_models):
    reg_best_r2 = max(regression_models, key=lambda item: item["metrics"]["r2"])
    reg_best_rmse = min(regression_models, key=lambda item: item["metrics"]["rmse"])
    cls_best_acc = max(classification_models, key=lambda item: item["metrics"]["accuracy"])
    cls_best_f1 = max(classification_models, key=lambda item: item["metrics"]["f1"])
    return [
        f"En regresión, el mejor R2 lo obtuvo {reg_best_r2['label']} con {reg_best_r2['metrics']['r2']:.4f}.",
        f"Si priorizas menor error de escala, el menor RMSE fue de {reg_best_rmse['label']} con {reg_best_rmse['metrics']['rmse']:.2f}.",
        f"En clasificación, el mejor accuracy fue de {cls_best_acc['label']} con {cls_best_acc['metrics']['accuracy']:.4f}.",
        f"Si buscas balance entre precisión y recall, el mejor F1 lo obtuvo {cls_best_f1['label']} con {cls_best_f1['metrics']['f1']:.4f}.",
    ]


def build_recommendations(regression_models, classification_models):
    reg_best = max(regression_models, key=lambda item: item["metrics"]["r2"])
    cls_best = max(classification_models, key=lambda item: item["metrics"]["accuracy"])
    return {
        "regression": {
            "model": reg_best["label"],
            "why": "Conviene cuando quieres la mejor explicación general del target numérico según R2 en esta corrida.",
        },
        "classification": {
            "model": cls_best["label"],
            "why": "Conviene cuando buscas mejor desempeño global para separar clases en este dataset.",
        },
    }


def run_optuna_summary(regression_dataset, classification_dataset):
    if optuna is None:
        return {
            "available": False,
            "title": "Optuna no disponible",
            "description": "Optuna es una librería para buscar hiperparámetros de forma automática y eficiente. Si la instalas, el tablero puede mostrar una búsqueda guiada de mejores configuraciones.",
        }

    reg_x_train, reg_x_test, reg_y_train, reg_y_test = split_dataset(regression_dataset)
    cls_x_train, cls_x_test, cls_y_train, cls_y_test = split_dataset(classification_dataset)

    def reg_objective(trial):
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 80, 180),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 6),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
            random_state=42,
        )
        model.fit(reg_x_train, reg_y_train)
        preds = model.predict(reg_x_test)
        return r2_score(reg_y_test, preds)

    def cls_objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 180),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 6),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
            random_state=42,
        )
        model.fit(cls_x_train, cls_y_train)
        preds = model.predict(cls_x_test)
        return accuracy_score(cls_y_test, preds)

    reg_study = optuna.create_study(direction="maximize")
    reg_study.optimize(reg_objective, n_trials=8, show_progress_bar=False)

    cls_study = optuna.create_study(direction="maximize")
    cls_study.optimize(cls_objective, n_trials=8, show_progress_bar=False)

    return {
        "available": True,
        "title": "Optuna",
        "description": "Optuna es una librería de optimización de hiperparámetros. En lugar de probar configuraciones manualmente, explora combinaciones de forma automática para encontrar mejores resultados con menos intentos.",
        "regression": {
            "metric": "R2",
            "value": round(reg_study.best_value, 4),
            "params": reg_study.best_params,
            "model": "Random Forest Regressor",
        },
        "classification": {
            "metric": "Accuracy",
            "value": round(cls_study.best_value, 4),
            "params": cls_study.best_params,
            "model": "Random Forest Classifier",
        },
    }


def detect_numeric_columns(rows, headers):
    numeric_columns = []
    for header in headers:
        is_numeric = True
        for row in rows[:40]:
            value = row.get(header, "").strip()
            if value == "":
                continue
            try:
                float(value)
            except ValueError:
                is_numeric = False
                break
        if is_numeric:
            numeric_columns.append(header)
    return numeric_columns


def parse_csv_upload(file_storage):
    raw_text = file_storage.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw_text))
    rows = [row for row in reader]
    headers = reader.fieldnames or []
    numeric_headers = detect_numeric_columns(rows, headers)
    return {
        "filename": file_storage.filename or "dataset.csv",
        "headers": headers,
        "rows": rows,
        "preview": rows[:8],
        "numeric_headers": numeric_headers,
    }


def build_dataset_from_csv(csv_state, dataset_kind, target_column, selected_features):
    rows = []
    for source_row in csv_state["rows"]:
        try:
            row = {feature: float(source_row.get(feature, "0") or 0) for feature in selected_features}
            target_value = float(source_row.get(target_column, "0") or 0)
            row["target"] = int(target_value) if dataset_kind == "classification" else float(target_value)
            rows.append(row)
        except ValueError:
            continue

    if not rows:
        raise ValueError("No se pudieron convertir filas válidas del CSV.")

    dataset_name = f"CSV Importado ({csv_state['filename']})"
    description = "Dataset cargado manualmente desde CSV y normalizado desde la interfaz."
    base = {
        "name": dataset_name,
        "description": description,
        "feature_names": selected_features,
        "rows": rows,
    }
    if dataset_kind == "classification":
        targets = sorted({int(row["target"]) for row in rows})
        base["target_names"] = [f"Clase {value}" for value in targets]
    return base


def predict_from_csv_rows(csv_state, dataset_kind):
    if dataset_kind == "regression":
        dataset = APP_STATE["regression_dataset"]
        models = APP_STATE["regression_models"]
    else:
        dataset = APP_STATE["classification_dataset"]
        models = APP_STATE["classification_models"]

    required_features = dataset["feature_names"]
    missing = [feature for feature in required_features if feature not in csv_state["headers"]]
    if missing:
        raise ValueError(
            "Al CSV le faltan columnas requeridas para predecir: " + ", ".join(missing)
        )

    parsed_rows = []
    for source_row in csv_state["rows"]:
        try:
            parsed_rows.append([float(source_row.get(feature, "0") or 0) for feature in required_features])
        except ValueError:
            continue

    if not parsed_rows:
        raise ValueError("No se encontraron filas válidas para generar predicciones.")

    prediction_tables = []
    for model_data in models:
        predictions = model_data["model"].predict(parsed_rows)
        preview = []
        for index, prediction in enumerate(predictions[:10], start=1):
            value = int(prediction) if dataset_kind == "classification" else round(float(prediction), 4)
            preview.append({"row": index, "prediction": value})
        prediction_tables.append(
            {
                "model": model_data["label"],
                "predictions": preview,
            }
        )

    return {
        "dataset_kind": dataset_kind,
        "record_count": len(parsed_rows),
        "required_features": required_features,
        "tables": prediction_tables,
    }


def predict_regression(manual_values):
    prediction_rows = []
    for model_data in APP_STATE["regression_models"]:
        prediction = model_data["model"].predict([manual_values])[0]
        prediction_rows.append({"label": model_data["label"], "prediction": round(float(prediction), 4)})
    return prediction_rows


def initialize_app_state():
    regression_dataset = load_regression_dataset()
    classification_dataset = load_classification_dataset()
    regression_models = train_regression_models(regression_dataset)
    classification_models = train_classification_models(classification_dataset)
    dimensionality = prepare_dimensionality_views(classification_dataset)
    hyperparameter_cards = build_hyperparameter_cards(regression_models + classification_models)
    conclusions = build_conclusions(regression_models, classification_models)
    recommendations = build_recommendations(regression_models, classification_models)
    optuna_summary = run_optuna_summary(regression_dataset, classification_dataset)

    APP_STATE.update(
        {
            "regression_dataset": regression_dataset,
            "classification_dataset": classification_dataset,
            "regression_models": regression_models,
            "classification_models": classification_models,
            "dimensionality": dimensionality,
            "hyperparameter_cards": hyperparameter_cards,
            "conclusions": conclusions,
            "recommendations": recommendations,
            "optuna_summary": optuna_summary,
            "r2_chart": make_plotly_bar(regression_models, "r2", "Regresión: comparación de R2", "#8b1e2a"),
            "rmse_chart": make_plotly_bar(regression_models, "rmse", "Regresión: comparación de RMSE", "#b69149"),
            "acc_chart": make_plotly_bar(classification_models, "accuracy", "Clasificación: comparación de accuracy", "#17324d"),
            "f1_chart": make_plotly_bar(classification_models, "f1", "Clasificación: comparación de F1", "#0f766e"),
            "csv_state": None,
            "csv_predictions": None,
            "model_notes": MODEL_NOTES,
        }
    )


def parse_manual_features(feature_names, form):
    return [float(form.get(feature, "0")) for feature in feature_names]


@app.route("/up-logo")
def up_logo():
    if UP_LOGO_PATH.exists():
        return send_file(UP_LOGO_PATH)
    return ("Logo no encontrado", 404)


@app.route("/", methods=["GET", "POST"])
def home():
    if not APP_STATE:
        initialize_app_state()

    active_tab = request.form.get("tab", "resumen") if request.method == "POST" else "resumen"
    prediction_rows = []
    regression_dataset = APP_STATE["regression_dataset"]
    manual_values = {feature: 0 for feature in regression_dataset["feature_names"]}
    message = ""
    csv_message = ""

    if request.method == "POST" and request.form.get("action") == "upload_csv":
        active_tab = "csv"
        csv_file = request.files.get("csv_file")
        if csv_file and csv_file.filename:
            try:
                APP_STATE["csv_state"] = parse_csv_upload(csv_file)
                APP_STATE["csv_predictions"] = None
                csv_message = "CSV cargado. Ahora puedes revisar columnas y ajustar el formato."
            except Exception:
                csv_message = "No se pudo leer el archivo CSV."
        else:
            csv_message = "Selecciona un archivo CSV antes de cargarlo."

    if request.method == "POST" and request.form.get("action") == "apply_csv_mapping":
        active_tab = "csv"
        csv_state = APP_STATE.get("csv_state")
        if csv_state:
            dataset_kind = request.form.get("dataset_kind", "regression")
            target_column = request.form.get("target_column", "")
            selected_features = request.form.getlist("feature_columns")
            if target_column and selected_features:
                try:
                    custom_dataset = build_dataset_from_csv(
                        csv_state, dataset_kind, target_column, selected_features
                    )
                    if dataset_kind == "regression":
                        APP_STATE["regression_dataset"] = custom_dataset
                        APP_STATE["regression_models"] = train_regression_models(custom_dataset)
                        APP_STATE["r2_chart"] = make_plotly_bar(APP_STATE["regression_models"], "r2", "Regresión: comparación de R2", "#8b1e2a")
                        APP_STATE["rmse_chart"] = make_plotly_bar(APP_STATE["regression_models"], "rmse", "Regresión: comparación de RMSE", "#b69149")
                    else:
                        APP_STATE["classification_dataset"] = custom_dataset
                        APP_STATE["classification_models"] = train_classification_models(custom_dataset)
                        APP_STATE["dimensionality"] = prepare_dimensionality_views(custom_dataset)
                        APP_STATE["acc_chart"] = make_plotly_bar(APP_STATE["classification_models"], "accuracy", "Clasificación: comparación de accuracy", "#17324d")
                        APP_STATE["f1_chart"] = make_plotly_bar(APP_STATE["classification_models"], "f1", "Clasificación: comparación de F1", "#0f766e")

                    APP_STATE["hyperparameter_cards"] = build_hyperparameter_cards(
                        APP_STATE["regression_models"] + APP_STATE["classification_models"]
                    )
                    APP_STATE["conclusions"] = build_conclusions(
                        APP_STATE["regression_models"], APP_STATE["classification_models"]
                    )
                    APP_STATE["recommendations"] = build_recommendations(
                        APP_STATE["regression_models"], APP_STATE["classification_models"]
                    )
                    APP_STATE["optuna_summary"] = run_optuna_summary(
                        APP_STATE["regression_dataset"], APP_STATE["classification_dataset"]
                    )
                    APP_STATE["csv_predictions"] = None
                    manual_values = {feature: 0 for feature in APP_STATE["regression_dataset"]["feature_names"]}
                    csv_message = "CSV aplicado correctamente al tablero."
                except ValueError as error:
                    csv_message = str(error)
            else:
                csv_message = "Selecciona una columna target y al menos una columna de features."

    if request.method == "POST" and request.form.get("action") == "predict_csv":
        active_tab = "csv"
        csv_state = APP_STATE.get("csv_state")
        if csv_state:
            try:
                dataset_kind = request.form.get("prediction_kind", "regression")
                APP_STATE["csv_predictions"] = predict_from_csv_rows(csv_state, dataset_kind)
                csv_message = (
                    f"Se detectaron {APP_STATE['csv_predictions']['record_count']} registros y se generaron predicciones para cada modelo."
                )
            except ValueError as error:
                APP_STATE["csv_predictions"] = None
                csv_message = str(error)
        else:
            csv_message = "Primero carga un CSV para poder generar predicciones."

    if request.method == "POST" and request.form.get("action") == "update_embeddings":
        active_tab = "reduccion"
        try:
            tsne_perplexity = float(request.form.get("tsne_perplexity", "30"))
            umap_neighbors = int(request.form.get("umap_neighbors", "15"))
            umap_min_dist = float(request.form.get("umap_min_dist", "0.1"))
            APP_STATE["dimensionality"] = prepare_dimensionality_views(
                APP_STATE["classification_dataset"],
                tsne_perplexity=max(5.0, min(tsne_perplexity, 60.0)),
                umap_neighbors=max(2, min(umap_neighbors, 80)),
                umap_min_dist=max(0.0, min(umap_min_dist, 0.99)),
            )
        except ValueError:
            message = "No se pudieron actualizar los hiperparámetros de t-SNE y UMAP."

    if request.method == "POST" and request.form.get("action") == "predict":
        active_tab = "prediccion"
        try:
            active_regression_dataset = APP_STATE["regression_dataset"]
            parsed_values = parse_manual_features(active_regression_dataset["feature_names"], request.form)
            manual_values = {feature: float(request.form.get(feature, "0")) for feature in active_regression_dataset["feature_names"]}
            prediction_rows = predict_regression(parsed_values)
            message = "Se calcularon predicciones para los modelos de regresión."
        except ValueError:
            message = "Introduce valores numéricos válidos en todas las variables."

    return render_template(
        "index.html",
        regression_dataset=APP_STATE["regression_dataset"],
        classification_dataset=APP_STATE["classification_dataset"],
        regression_models=APP_STATE["regression_models"],
        classification_models=APP_STATE["classification_models"],
        active_tab=active_tab,
        prediction_rows=prediction_rows,
        manual_values=manual_values,
        message=message,
        csv_message=csv_message,
        up_logo_exists=UP_LOGO_PATH.exists(),
        umap_available=APP_STATE["dimensionality"]["umap_available"],
        dimensionality_params=APP_STATE["dimensionality"]["params"],
        r2_chart=APP_STATE["r2_chart"],
        rmse_chart=APP_STATE["rmse_chart"],
        acc_chart=APP_STATE["acc_chart"],
        f1_chart=APP_STATE["f1_chart"],
        prediction_chart=make_plotly_prediction(prediction_rows),
        pca_chart=APP_STATE["dimensionality"]["pca_chart"],
        tsne_chart=APP_STATE["dimensionality"]["tsne_chart"],
        umap_chart=APP_STATE["dimensionality"]["umap_chart"],
        hyperparameter_cards=APP_STATE["hyperparameter_cards"],
        conclusions=APP_STATE["conclusions"],
        recommendations=APP_STATE["recommendations"],
        optuna_summary=APP_STATE["optuna_summary"],
        csv_state=APP_STATE["csv_state"],
        csv_predictions=APP_STATE["csv_predictions"],
        team_members=TEAM_MEMBERS,
        professor_name=PROFESSOR_NAME,
        model_notes=APP_STATE["model_notes"],
    )


initialize_app_state()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
