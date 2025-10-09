# -*- coding: utf-8 -*-

import os
os.environ["MPLBACKEND"] = "Agg"   # backend non-GUI pour matplotlib
import sys
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from mlflow.models.signature import infer_signature

# === Imports de tes modules existants ===
try:
    from src import data_pipeline as dp      # get_config, load_data, split_data, eda_quick
    from src import modeling as mod          # build_preprocessor, make_*_pipeline, train_validate_simple, evaluate_on_test
except ImportError:
    import data_pipeline as dp
    import modeling as mod

# === Paramètres globaux ===
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 19), 2)
RECALL_MIN, PREC_MIN = 0.65, 0.30
ARTIFACTS_DIR = "artifacts_modèles"
EXPERIMENT_NAME = "loan_default_v2"

def log_model_with_signature(pipeline, X_train, model_name, cfg):
    """
    Crée un input_example réaliste, cast les colonnes entières en float64,
    génère la signature et loggue le pipeline dans le run courant.
    """
    sample_n = min(200, len(X_train))
    input_example = X_train.sample(sample_n, random_state=cfg.get("seed", 42)).copy()

    # Cast de toutes les colonnes entières vers float64 pour éviter le warning MLflow
    int_like = input_example.select_dtypes(include=[
        "int64", "int32", "Int64", "uint8", "UInt8", "UInt16", "UInt32", "UInt64"
    ]).columns
    if len(int_like) > 0:
        input_example[int_like] = input_example[int_like].astype("float64")

    signature = infer_signature(input_example, pipeline.predict_proba(input_example)[:, 1])

    # MLflow récents préfèrent "name"; fallback sur "artifact_path" si besoin
    try:
        mlflow.sklearn.log_model(
            pipeline,
            name=f"{model_name}_pipeline",
            signature=signature,
            input_example=input_example
        )
    except TypeError:
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=f"model_{model_name}",
            signature=signature,
            input_example=input_example
        )

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:8086")

    # 0) Config + EDA
    cfg = dp.get_config()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    try:
        df = dp.load_data(cfg)
    except FileNotFoundError as e:
        print(f"[ERREUR] {e}")
        sys.exit(1)

    dp.eda_quick(df, cfg, outdir=ARTIFACTS_DIR)

    # 1) Split
    X_train, X_val, X_test, y_train, y_val, y_test = dp.split_data(df, cfg)
    num_cols = [c for c in cfg.get("nums_cols", []) if c in X_train.columns]

    # 2) Préprocesseurs adaptés (linéaire vs arbres)
    pre_linear = mod.build_preprocessor(num_cols, for_linear=True)
    pre_tree   = mod.build_preprocessor(num_cols, for_linear=False)

    # 3) Pipelines modèles (fabriques de TON modeling.py)
    models = {
        "logreg": mod.make_logreg_pipeline(pre_linear),
        "tree":   mod.make_tree_pipeline(pre_tree, max_depth=None, min_samples_leaf=1),
        "rf":     mod.make_rf_pipeline(pre_tree, n_estimators=300, max_depth=None, min_samples_leaf=1),
    }

    # 4) MLflow : configure l’expérience et coupe l’autolog (tu logs à la main)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(disable=True)

    # === Un run par modèle AU 1ER NIVEAU (pas de run parent) ===
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name, nested=False):
            mlflow.set_tags({"author": "Diallo", "stage": "dev", "model": name})
            mlflow.log_params({"test_ratio": cfg["test_size"], "val_ratio": cfg["val_size"]})

            # === Validation (entraîne, choisit le seuil, sauve artefacts val) ===
            res_val = mod.train_validate_simple(
                pipeline=pipe,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                threshold_grid=THRESHOLD_GRID,
                recall_min=RECALL_MIN, prec_min=PREC_MIN,
                artifacts_modèles_dir=ARTIFACTS_DIR,
                prefix=f"{name}_val"
            )

            # CSV des seuils (déjà calculés) → loggue comme artefact
            thr_df = pd.DataFrame(res_val["threshold_table"])
            thr_csv = os.path.join(ARTIFACTS_DIR, f"{name}_val_thresholds.csv")
            thr_df.to_csv(thr_csv, index=False)
            mlflow.log_artifact(thr_csv)

            # Métriques validation
            vg, at = res_val["val_global"], res_val["val_at_thr"]
            mlflow.log_metrics({
                "val_roc_auc": vg["roc_auc"],
                "val_pr_auc":  vg["pr_auc"],
                "val_logloss": vg["logloss"],
                "val_precision_at_thr": at["precision"],
                "val_recall_at_thr":    at["recall"],
                "val_f1_at_thr":        at["f1"],
                "val_thr":              res_val["best_threshold"],
            })

            # Images validation (ROC/PR/CM)
            for suf in ("roc", "pr", "cm"):
                p = os.path.join(ARTIFACTS_DIR, f"{name}_val_{suf}.png")
                if os.path.exists(p):
                    mlflow.log_artifact(p)

            # === Test (évalue @ seuil choisi, sauve artefacts test) ===
            with mlflow.start_run(run_name=f"{name}_test", nested=True):
                test_metrics = mod.evaluate_on_test(
                    pipeline=res_val["pipeline"],
                    X_test=X_test, y_test=y_test,
                    threshold=float(res_val["best_threshold"]),
                    artifacts_modèles_dir=ARTIFACTS_DIR,
                    prefix=f"{name}_test"
                )
                mlflow.log_metrics(test_metrics)
                for suf in ("roc", "pr", "cm"):
                    p = os.path.join(ARTIFACTS_DIR, f"{name}_test_{suf}.png")
                    if os.path.exists(p):
                        mlflow.log_artifact(p)

            # === Log du pipeline de CE modèle (aucune sélection “meilleur”) ===
            log_model_with_signature(res_val["pipeline"], X_train, model_name=name, cfg=cfg)

    print("\nTerminé. Tous les modèles ont été loggués.")
    print(f"Ouvre MLflow UI (mlflow ui) et va dans l'expérience {EXPERIMENT_NAME}.")
    print("Tu verras des runs au 1er niveau: logreg, tree, rf (chacun avec un run enfant *_test).")
    print(f"Artefacts enregistrés dans: {os.path.abspath(ARTIFACTS_DIR)}")

if __name__ == "__main__":
    main()
