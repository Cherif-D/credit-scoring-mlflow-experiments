# ce fichier a pour but de charger la data, faire le splitting, l'eda et le preprocessing

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


# Fichier de configuration :
config = {
    "data_path": "Data/Loan_Data.csv",
    "Target" : "default",
    
    # colonnes numériques
    "nums_cols" : ["credit_lines_outstanding", "fico_score",
                   "income", "loan_amt_outstanding",
                   "total_debt_outstanding","years_employed"],
    
    # Colonnes à exclure
    "Drop_cols" : ["customer_id"],
    
    # découpage train test
    "test_size" : 0.2,
    "val_size" : 0.25,
    "seed" : 42,
    
    # Paramètres cross validation
    "cv_N_splits" : 5, # nombre de folds
    "cv_shuffle" : True, # mélanger les données avant de faire les folds
    }

def get_config():
    """Retourne la configuration du projet sous forme de dictionnaire."""
    return config


def load_data(cfg):
    "Charge les données à partir du chemin spécifié dans la configuration."
    base_dir = Path(__file__).resolve().parents[1]   # <- dossier ML_FLOW (parent de src)
    data_path = Path(cfg["data_path"])
    if not data_path.is_absolute():
        data_path = (base_dir / data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {data_path}\n"
            f"Astuce: vérifie l'emplacement ou adapte config['data_path']."
        )

    df = pd.read_csv(data_path)
    assert cfg["Target"] in df.columns, f"La colonne cible {cfg['Target']} n'est pas dans les données"
    return df



# Splitting : splitting en train, test et validation

def split_data(df, cfg):
    """Decoupage en train/val/test de façon stratifiée (sans cross validation pour l'instant)
    Stratifiée : on conserve la même proportion de chaque classe dans chaque partie que dans le dataset initial"""
    
    y = df[cfg["Target"]].astype('int')  # s'assurer que y est de type int
    X = df.drop(columns=[cfg["Target"]]) # toutes les colonnes sauf la cible
    
    # premier split : train+val et test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"] )
    
    val_ratio = cfg["val_size"] / (1 - cfg["test_size"]) # ajuster la taille de validation par rapport à train+val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=cfg["seed"] )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# splitting en prenant en compte la cross_validation sur le train

def split_trainval_test_for_cv(df, cfg):
    """Separe en train_val et une partie test
    On evalue tout seul le modèle sur la partie test
    On fait la sélection de modèle/hyperparamètres via la cross validation sur trainval
    
    """
    
    y = df[cfg["Target"]].astype(int) 
    X = df.drop(columns=[cfg["Target"]])  
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=y
    )
    return X_trainval, X_test, y_trainval, y_test

def get_cv_splitter(cfg):
    """Retourne un objet de cross-validation basé sur la configuration."""
  
    cv_splitter = StratifiedKFold(
        n_splits=cfg["cv_N_splits"],
        shuffle=cfg["cv_shuffle"],
        random_state=cfg["seed"]
    )
    return cv_splitter

# EDA : analyse exploratoire des données (visualisation des distributions, corrélations, etc.)

# EDA “quick”
# =========================
def eda_quick(df: pd.DataFrame, cfg: dict, outdir="artifacts"):
    """
    - Summary num (summary.xlsx)
    - Balance de la cible (class_balance.csv + target_dist.png)
    - Top NA (top_missing.csv + top_missing.png)
    - Corrélations numériques (corr_num.csv + corr_num.png)
    - Distributions numériques (dist_num_<col>.png)
    """
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    target = cfg["Target"]
    num_cols = cfg.get("nums_cols", [])

    # Filtrer les colonnes numériques réellement présentes (évite les KeyError)
    present_num = [c for c in num_cols if c in df.columns]
    missing_num = [c for c in num_cols if c not in df.columns]
    if missing_num:
        print("[EDA][WARN] Colonnes numériques manquantes ignorées :", missing_num)

    # 1) Summary (numérique)
    summary_num = df[present_num].describe().T if present_num else pd.DataFrame()
    with pd.ExcelWriter(out / "summary.xlsx") as w:
        if not summary_num.empty:
            summary_num.to_excel(w, sheet_name="numeric")

    # 2) Balance de classe
    counts = df[target].value_counts().sort_index()
    class_balance = pd.DataFrame({
        "class": counts.index, "count": counts.values,
        "ratio": counts.values / counts.values.sum()
    })
    class_balance.to_csv(out / "class_balance.csv", index=False)
    counts.plot(kind="bar")
    plt.title("Distribution de la cible"); plt.xlabel(target); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out / "target_dist.png"); plt.close()

    # 3) Top NA
    na = df.isna().sum().sort_values(ascending=False)
    na.head(20).to_csv(out / "top_missing.csv", header=["n_missing"])
    na.head(20).plot(kind="bar")
    plt.title("Top colonnes avec valeurs manquantes"); plt.ylabel("Nombre de NA")
    plt.tight_layout(); plt.savefig(out / "top_missing.png"); plt.close()

    # 4) Corrélations numériques
    if present_num:
        corr = df[present_num].corr(numeric_only=True)
        corr.to_csv(out / "corr_num.csv")
        plt.figure(); plt.imshow(corr, interpolation="nearest")
        plt.xticks(range(len(present_num)), present_num, rotation=90)
        plt.yticks(range(len(present_num)), present_num); plt.colorbar()
        plt.title("Corrélation (numériques)"); plt.tight_layout()
        plt.savefig(out / "corr_num.png"); plt.close()

    # 5) Distributions numériques
    for col in present_num:
        s = df[col].dropna()
        if s.empty:
            continue
        plt.figure(); plt.hist(s, bins=30)
        plt.title(f"Distribution – {col}"); plt.xlabel(col); plt.ylabel("Fréquence")
        plt.tight_layout(); plt.savefig(out / f"dist_num_{col}.png"); plt.close()

    return {
        "summary_xlsx": str(out / "summary.xlsx") if not summary_num.empty else None,
        "class_balance_csv": str(out / "class_balance.csv"),
        "target_dist_png": str(out / "target_dist.png"),
        "top_missing_csv": str(out / "top_missing.csv"),
        "top_missing_png": str(out / "top_missing.png"),
        "corr_num_csv": str(out / "corr_num.csv") if present_num else None,
        "corr_num_png": str(out / "corr_num.png") if present_num else None,
    }



# utilisation :
if __name__ == "__main__":
    cfg = get_config()
    df = load_data(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)
    artifacts = eda_quick(df, cfg, outdir="artifacts")
    print(artifacts)
    print("EDA artifacts saved in 'artifacts/' directory.")  