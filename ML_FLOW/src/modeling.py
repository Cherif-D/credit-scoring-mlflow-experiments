# Dans ce fichier nous allons essayer de faire la modélisation :
# On prend pour baseline la regression logistique et on essaye de tester
# d'autres modèles comme decision tree, random forest.
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from typing import Dict, List


# # === Préprocesseur simple (pas de NA) ===
def build_preprocessor(num_cols: List[str], for_linear: bool = True):
    """
    - Pour Régression Logistique: StandardScaler sur numériques
    - Pour Arbres/Forêt: passthrough (les arbres n'ont pas besoin de scaling)
    """
    if not num_cols:
        return "passthrough"
    if for_linear:
        return ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")
    else:
        return ColumnTransformer([("num", "passthrough", num_cols)], remainder="drop")


# fonction pour les metriques globales(sans seuil)
def global_metrics(y_true, y_pred_proba) -> Dict[str, float]:
    """Calcule les métriques globales pour les prédictions probabilistes.

    Args:
        y_true (array-like): Vraies étiquettes.
        y_pred_proba (array-like): Probabilités prédites pour la classe positive.

    Returns:
        Dict[str, float]: Dictionnaire contenant les métriques calculées.
    """
    

    auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)
    return {"roc_auc": auc, "pr_auc": pr_auc, "logloss": logloss}

def sweep_thresholds(y_true, y_proba, thr_list: List[float]) -> List[Dict[str, float]]:
    """Teste plusieurs seuils et calcule précision, rappel et F1 pour chacun.
    Utile pour tracer la courbe précision–rappel.
    En changeant le seuil, on choisit le meilleur compromis au lieu de fixer 0,5."""
    rows = []
    for thr in thr_list:
        y_pred = (y_proba >= thr).astype(int)
        rows.append({
            "thr": float(thr),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0)
        })
    return rows

def choose_threshold(rows: List[Dict[str, float]],
                     recall_min: float = 0.65,
                     prec_min: float = 0.30) -> Dict[str, float]:
    """
    Objectif (en français simple) :
    -------------------------------
    - On a testé plein de seuils (rows = liste de dicts avec "thr", "precision", "recall", "f1").
    - On veut choisir un seuil "utilisable en vrai" :
        1) D'abord, on garde seulement les seuils qui respectent des MINIMA métier :
           - rappel (recall) >= recall_min
           - précision (precision) >= prec_min
        2) Parmi ces candidats, on prend celui qui a le meilleur F1.
        3) S'il n'y a AUCUN candidat (personne ne passe les minima),
           on prend quand même le seuil qui a le meilleur F1 global (fallback).

    Paramètres :
    - rows : résultats du balayage des seuils, ex. [{"thr": 0.5, "precision": ..., "recall": ..., "f1": ...}, ...]
    - recall_min : rappel minimum acceptable (ex. 0.65 = 65%)
    - prec_min   : précision minimum acceptable (ex. 0.30 = 30%)

    Retour :
    - Un dictionnaire correspondant au "meilleur" seuil selon ces règles.
    """

    # Sécurité : si la liste est vide, on lève une erreur explicite
    if not rows:
        raise ValueError("La liste 'rows' est vide : aucun seuil à choisir.")

    # 1) Filtre "garde-fous" : on ne garde que les seuils qui passent les minima
    cand = [r for r in rows if r.get("recall", 0.0) >= recall_min and r.get("precision", 0.0) >= prec_min]

    # 2) Si on a des candidats, on choisit dedans ; sinon, on se rabat sur tous les seuils
    base = cand if len(cand) > 0 else rows

    # 3) On prend l'élément dont le F1 est maximal
    #    key=lambda r: r["f1"] signifie "compare les éléments selon la clé 'f1'"
    return max(base, key=lambda r: r.get("f1", -np.inf))


def eval_at_threshold(y_true, y_proba, thr: float) -> Dict[str, float]:
    """
    Objectif (en français simple) :
    -------------------------------
    - Avec un seuil donné 'thr', on transforme les probabilités en classes 0/1.
    - On calcule ensuite :
        * précision, rappel, F1
        * la matrice de confusion (TN, FP, FN, TP) pour comprendre les erreurs.

    Paramètres :
    - y_true  : vraies étiquettes (0/1), array-like
    - y_proba : probabilités prédites (entre 0 et 1), array-like
    - thr     : seuil choisi (ex. 0.5)

    Retour :
    - Un dict avec precision, recall, f1, tn, fp, fn, tp
    """

    # Sécurité : conversion en arrays NumPy (utile si on reçoit des listes Python)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    # On transforme la proba en classe : 1 si proba >= thr, sinon 0
    y_pred = (y_proba >= float(thr)).astype(int)

    # On récupère TN, FP, FN, TP depuis la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # On calcule les métriques de base ; zero_division=0 évite les erreurs quand aucun positif n'est prédit
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "thr": float(thr)  # pratique pour tracer ou logger
    }
    
    
def save_plots(y_true, y_proba, y_pred, outdir: str, prefix: str) -> None:
    """Sauvegarde ROC/PR/Confusion Matrix dans artifacts_modèles/."""
    os.makedirs(outdir, exist_ok=True)  # crée le dossier s’il n’existe pas

    # 1) Courbe ROC
    RocCurveDisplay.from_predictions(y_true, y_proba)  # calcule/trace ROC à partir des proba
    plt.title(f"ROC - {prefix}")                      # titre du graphique
    plt.tight_layout(); plt.savefig(f"{outdir}/{prefix}_roc.png"); plt.close()

    # 2) Courbe Precision-Recall
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)  # calcule/trace PR
    plt.title(f"PR - {prefix}")
    plt.tight_layout(); plt.savefig(f"{outdir}/{prefix}_pr.png"); plt.close()

    # 3) Matrice de confusion (nécessite des prédictions binaires)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)   # calcule/trace CM
    plt.title(f"CM - {prefix}")
    plt.tight_layout(); plt.savefig(f"{outdir}/{prefix}_cm.png"); plt.close()

# Les fonctions de modélisation :

def make_logreg_pipeline(preprocessor) -> Pipeline:
    """Baseline : Logistic Regression (robuste classes déséquilibrées)."""
    clf = LogisticRegression(class_weight="balanced", max_iter=300, C=1.0)
    return Pipeline([("pre", preprocessor), ("clf", clf)])

def make_tree_pipeline(preprocessor, max_depth=None, min_samples_leaf=1) -> Pipeline:
    """DecisionTree : simple, interprétable ; bons benchmarks rapides."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight="balanced", random_state=42
    )
    return Pipeline([("pre", preprocessor), ("clf", clf)])

def make_rf_pipeline(preprocessor, n_estimators=300, max_depth=None, min_samples_leaf=1) -> Pipeline:
    """RandomForest : souvent très bon par défaut, robuste ; baseline forte."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    )
    return Pipeline([("pre", preprocessor), ("clf", clf)])


# Mode split classique train/val/test (sans cross-validation)

def train_validate_simple(pipeline: Pipeline,
                          X_train, y_train, X_val, y_val,
                          threshold_grid=(0.1,0.2,0.3,0.4,0.5),
                          recall_min=0.65, prec_min=0.30,
                          artifacts_modèles_dir="artifacts_modèles",
                          prefix="val") -> Dict[str, object]:
    """
    Fit sur TRAIN, choisit un seuil optimal sur VAL, sort métriques & plots VAL.
    Retourne le pipeline entraîné + le seuil choisi + métriques VAL.
    """
    # 1) Fit
    pipeline.fit(X_train, y_train)

    # 2) Probas sur VAL
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    # 3) Métriques globales (ROC/PR)
    gm = global_metrics(y_val, y_val_proba)

    # 4) Choix seuil sur VAL
    rows = sweep_thresholds(y_val, y_val_proba, threshold_grid)
    chosen = choose_threshold(rows, recall_min, prec_min)
    thr = float(chosen["thr"])

    # 5) Plots & metrics @ seuil
    y_val_pred = (y_val_proba >= thr).astype(int)
    save_plots(y_val, y_val_proba, y_val_pred, artifacts_modèles_dir, prefix=prefix)
    at_thr = eval_at_threshold(y_val, y_val_proba, thr)

    return {
        "pipeline": pipeline,
        "val_global": gm,
        "val_at_thr": at_thr,
        "best_threshold": thr,
        "threshold_table": rows
    }

def evaluate_on_test(pipeline: Pipeline,
                     X_test, y_test,
                     threshold: float,
                     artifacts_modèles_dir="artifacts_modèles",
                     prefix="test") -> Dict[str, float]:
    """
    Évalue le pipeline final sur TEST à un seuil donné + sauvegarde plots TEST.
    """
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    gm = global_metrics(y_test, y_test_proba)
    at_thr = eval_at_threshold(y_test, y_test_proba, threshold)

    y_test_pred = (y_test_proba >= threshold).astype(int)
    save_plots(y_test, y_test_proba, y_test_pred, artifacts_modèles_dir, prefix=prefix)

    return {
        "test_roc_auc": gm["roc_auc"],
        "test_pr_auc":  gm["pr_auc"],
        "test_precision_at_thr": at_thr["precision"],
        "test_recall_at_thr":    at_thr["recall"],
        "test_f1_at_thr":        at_thr["f1"],
        "thr_used": float(threshold)
    }


# Mode split avec cross-validation sur train/val

def cross_validate(pipeline_template: Pipeline,
                   X_trainval, y_trainval,
                   cv_splitter,
                   threshold_grid=(0.1,0.2,0.3,0.4,0.5),
                   recall_min=0.65, prec_min=0.30,
                   artifacts_modèles_dir="artifacts_modèles",
                   run_prefix="cv_fold") -> Dict[str, object]:
    """
    Effectue une CV (StratifiedKFold) sur le bloc trainval :
      - à chaque fold : fit sur train_fold, seuil choisi sur val_fold
      - mesures globales + @ seuil loggées par fold
      - plots sauvegardés par fold (ROC/PR/CM)

    Retour :
      - moyennes ROC/PR et Precision/Recall/F1 @ seuil
      - liste des seuils par fold + suggéré (médiane)
    """
    fold_globals, fold_at_thr, fold_thr = [], [], []

    os.makedirs(artifacts_modèles_dir, exist_ok=True)

    for k, (tr_idx, va_idx) in enumerate(cv_splitter.split(X_trainval, y_trainval), start=1):
        # 1) Split fold
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]

        # 2) Cloner un pipeline "propre" pour le fold
        pipe = clone(pipeline_template)

        # 3) Fit sur train_fold
        pipe.fit(X_tr, y_tr)

        # 4) Probas sur val_fold
        y_va_proba = pipe.predict_proba(X_va)[:, 1]

        # 5) Global metrics + seuil fold
        gm = global_metrics(y_va, y_va_proba)
        rows = sweep_thresholds(y_va, y_va_proba, threshold_grid)
        chosen = choose_threshold(rows, recall_min, prec_min)
        thr_k = float(chosen["thr"])

        # 6) Metrics @ seuil + plots
        y_va_pred = (y_va_proba >= thr_k).astype(int)
        save_plots(y_va, y_va_proba, y_va_pred, artifacts_modèles_dir, prefix=f"{run_prefix}_{k}")

        at_thr = eval_at_threshold(y_va, y_va_proba, thr_k)

        # 7) Empiler résultats du fold
        fold_globals.append(gm)
        fold_at_thr.append(at_thr)
        fold_thr.append(thr_k)

    # 8) Agréger sur les folds
    avg_global = {
        "roc_auc": float(np.mean([g["roc_auc"] for g in fold_globals])),
        "pr_auc":  float(np.mean([g["pr_auc"]  for g in fold_globals]))
    }

    avg_at_thr = {
        "precision": float(np.mean([a["precision"] for a in fold_at_thr])),
        "recall":    float(np.mean([a["recall"]    for a in fold_at_thr])),
        "f1":        float(np.mean([a["f1"]        for a in fold_at_thr]))
    }
    suggested_thr = float(np.median(fold_thr)) if len(fold_thr) else 0.5

    return {
        "cv_avg_global": avg_global,
        "cv_avg_at_thr": avg_at_thr,
        "cv_thresholds": fold_thr,
        "cv_suggested_threshold": suggested_thr
    }
    
    
# Script d'utilisation



def _safe_get_feature_names(preprocessor) -> List[str]:
    """Essaie de récupérer les noms de features sortant du ColumnTransformer."""
    try:
        fn = preprocessor.get_feature_names_out()
        return list(fn)
    except Exception:
        return []

def _export_feature_importances(pipe: Pipeline, model_name: str, outdir: str) -> None:
    """Exporte les importances (si dispo) pour les modèles type arbre/forêt."""
    try:
        clf = pipe.named_steps.get("clf")
        pre = pipe.named_steps.get("pre")
        if hasattr(clf, "feature_importances_"):
            names = _safe_get_feature_names(pre) or [f"feat_{i}" for i in range(len(clf.feature_importances_))]
            imps = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
            imps.sort_values("importance", ascending=False, inplace=True)
            os.makedirs(outdir, exist_ok=True)
            imps.to_csv(os.path.join(outdir, f"feat_importances_{model_name}.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Impossible d'exporter les importances pour {model_name}: {e}")

def run_all_models():
    # 0) Charger data_pipeline
    try:
        import data_pipeline as dp
    except Exception as e:
        print("[ERREUR] Impossible d'importer data_pipeline.py :", e); sys.exit(1)

    # 1) Config + données + splits
    cfg = dp.get_config()
    df  = dp.load_data(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = dp.split_data(df, cfg)

    # 2) Préprocesseurs
    num_cols = [c for c in cfg.get("nums_cols", []) if c in X_train.columns]
    pre_logreg = build_preprocessor(num_cols, for_linear=True)   # StandardScaler
    pre_tree   = build_preprocessor(num_cols, for_linear=False)  # passthrough

    # 3) Modèles
    models = {
        "logreg": make_logreg_pipeline(pre_logreg),
        "tree":   make_tree_pipeline(pre_tree, max_depth=None, min_samples_leaf=1),
        "rf":     make_rf_pipeline(pre_tree, n_estimators=300, max_depth=None, min_samples_leaf=1),
    }

    thr_grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    artifacts_modèles_dir = "artifacts_modèles"; os.makedirs(artifacts_modèles_dir, exist_ok=True)

    summary_rows = []
    for name, pipe in models.items():
        print(f"\n===== Modèle: {name} =====")
        res_val = train_validate_simple(
            pipe, X_train, y_train, X_val, y_val,
            threshold_grid=thr_grid, recall_min=0.65, prec_min=0.30,
            artifacts_modèles_dir=artifacts_modèles_dir, prefix=f"{name}_val"
        )

        # Export table des seuils
        pd.DataFrame(res_val["threshold_table"]).to_csv(
            os.path.join(artifacts_modèles_dir, f"{name}_val_thresholds.csv"), index=False
        )

        # Éval test au seuil choisi
        best_thr = float(res_val["best_threshold"])
        test_metrics = evaluate_on_test(
            res_val["pipeline"], X_test, y_test, threshold=best_thr,
            artifacts_modèles_dir=artifacts_modèles_dir, prefix=f"{name}_test"
        )

        vg, at = res_val["val_global"], res_val["val_at_thr"]
        summary_rows.append({
            "model": name, "best_threshold": best_thr,
            "val_roc_auc": vg["roc_auc"], "val_pr_auc": vg["pr_auc"], "val_logloss": vg["logloss"],
            "val_precision_at_thr": at["precision"], "val_recall_at_thr": at["recall"], "val_f1_at_thr": at["f1"],
            "val_TN": at["tn"], "val_FP": at["fp"], "val_FN": at["fn"], "val_TP": at["tp"],
            "test_roc_auc": test_metrics["test_roc_auc"], "test_pr_auc": test_metrics["test_pr_auc"],
            "test_precision_at_thr": test_metrics["test_precision_at_thr"],
            "test_recall_at_thr": test_metrics["test_recall_at_thr"],
            "test_f1_at_thr": test_metrics["test_f1_at_thr"],
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["test_f1_at_thr", "val_f1_at_thr", "val_pr_auc"], ascending=False
    )
    summary_path = os.path.join(artifacts_modèles_dir, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n================= RÉCAPITULATIF =================")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary)
    print(f"\n-> CSV récap: {summary_path}")
    
if __name__ == "__main__":
    run_all_models()


