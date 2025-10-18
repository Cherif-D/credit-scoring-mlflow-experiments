# app.py — Loan Default Scoring (UI sans presets)

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ============================ Paramètres ============================

MODEL_PKL  = os.getenv("MODEL_PKL", "model.pkl")
DATA_PATH  = os.getenv("DATA_PATH", "data/Loan_Data.csv")
TARGET_COL = os.getenv("TARGET_COL", "default")
DEF_THRESH = float(os.getenv("THRESHOLD", "0.5"))
PRIMARY_COLOR = os.getenv("PRIMARY_COLOR", "#4f46e5")  # indigo-600 par défaut

st.set_page_config(page_title="Loan Default — Scoring", page_icon="🏦", layout="wide")

# Colonnes sans seuil (min/max)
NO_BOUNDS: set[str] = {
    "total_debt_outstanding",
    "income",
    "years_employed",
    "loan_amt_outstanding",
    "loan_amt_oustanding",  # tolérance orthographe
}

# Contraintes indispensables (min, max) ; None = pas de borne
HARD_BOUNDS: Dict[str, Tuple[float | None, float | None]] = {
    "fico_score": (300, 850),
    "credit_lines_outstanding": (0, None),
    "customer_id": (0, None),
}

# Colonnes entières (si on n'a pas le CSV pour le déduire)
LIKELY_INT_COLS: set[str] = {
    "credit_lines_outstanding", "years_employed", "fico_score", "customer_id"
}

# Groupes UI (facultatif si la colonne est absente du modèle)
GROUPS = {
    "Identité": ["customer_id"],
    "Encours": ["credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding"],
    "Revenus & Emploi": ["income", "years_employed"],
    "Score": ["fico_score"],
}

# ============================ Style ============================

st.markdown(
    f"""
    <style>
      #MainMenu {{visibility:hidden;}}
      footer {{visibility:hidden;}}

      .app-card {{
        border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px;
        background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      }}
      .pill {{
        display:inline-block; padding:4px 10px; border-radius:999px; color:white;
        font-weight:600; font-size:0.9rem;
      }}
      .pill-ok {{ background:#10b981; }}    /* emerald-500 */
      .pill-bad {{ background:#ef4444; }}   /* red-500 */

      .badge {{
        display:inline-block; padding:2px 8px; border-radius:8px;
        background:#eef2ff; color:#3730a3; font-weight:600; font-size:0.8rem;
      }}

      .primary {{ color: {PRIMARY_COLOR}; }}
      .btn-primary > button {{ background:{PRIMARY_COLOR} !important; color:#fff !important; border-radius:12px !important; }}
      .btn-ghost   > button {{ border:1px solid #e5e7eb !important; border-radius:12px !important; background:white !important; color:#111827 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================ Utilitaires ============================

@st.cache_resource(show_spinner=False)
def load_model(pkl_path: str):
    p = Path(pkl_path)
    if not p.exists():
        base = Path(__file__).resolve().parent
        for cand in [(base / pkl_path).resolve(), (base / ".." / pkl_path).resolve()]:
            if cand.exists():
                p = cand; break
    if not p.exists():
        raise FileNotFoundError(f"Modèle introuvable: {pkl_path}")
    return joblib.load(p), str(p)

def try_load_df(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        base = Path(__file__).resolve().parent
        for cand in [(base / path).resolve(), (base / ".." / path).resolve()]:
            if cand.exists():
                p = cand; break
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def parse_float_maybe(raw: str) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        pass
    try:
        from scipy.special import expit
        return expit(model.decision_function(X))
    except Exception:
        y = np.asarray(model.predict(X), dtype=float)
        return y[:, 1] if y.ndim == 2 and y.shape[1] > 1 else y

def draw_gauge(prob: float):
    """Jauge circulaire (donut) simple avec matplotlib."""
    prob = max(0.0, min(1.0, prob))
    sizes = [prob, 1 - prob]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    wedges, _ = ax.pie(
        sizes, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor="white")
    )
    wedges[0].set_facecolor(PRIMARY_COLOR)
    wedges[1].set_facecolor("#e5e7eb")
    ax.text(0, 0, f"{prob:.3f}\n", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set(aspect="equal")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

def defaults_from_df(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            d[c] = float(np.nanmedian(df[c].dropna())) if df[c].notna().any() else 0.0
        else:
            d[c] = ""
    return d

# ============================ Chargements ============================

try:
    model, resolved_model_path = load_model(MODEL_PKL)
except Exception as e:
    st.error(f"❌ Impossible de charger le modèle: {e}")
    st.stop()

df = try_load_df(DATA_PATH)

# Features dans l'ordre du modèle si dispo
if hasattr(model, "feature_names_in_") and len(getattr(model, "feature_names_in_")) > 0:
    feature_cols: List[str] = list(getattr(model, "feature_names_in_"))
else:
    if df is None:
        st.error("❌ Ni dataset ni feature_names_in_ : impossible de déduire les variables d'entrée.")
        st.stop()
    if TARGET_COL not in df.columns:
        st.error(f"❌ Colonne cible « {TARGET_COL} » absente du CSV.")
        st.stop()
    feature_cols = [c for c in df.columns if c != TARGET_COL]

# ============================ En-tête ============================

c1, c2 = st.columns([0.72, 0.28])
with c1:
    st.markdown(f"<h2 class='primary'>🏦 Loan Default — Scoring</h2>", unsafe_allow_html=True)
    st.caption("Saisis librement tes valeurs (virgule ou point). Seules quelques variables ont des bornes logiques (FICO, comptes ≥ 0).")
with c2:
    st.markdown("<div class='app-card'><span class='badge'>Modèle</span><br/>"
                f"<code>{Path(resolved_model_path).name}</code><br/>"
                f"{'Dataset : <code>'+Path(DATA_PATH).name+'</code>' if df is not None else 'Dataset : <i>non fourni</i>'}"
                "</div>", unsafe_allow_html=True)

st.divider()

# ============================ Seuil uniquement (pas de presets) ============================

if "history" not in st.session_state:
    st.session_state["history"] = []

# valeurs par défaut (CSV -> médianes, sinon 0)
default_inputs = defaults_from_df(df, feature_cols) if df is not None else {c: 0.0 for c in feature_cols}

# Slider seuil aligné à droite
tool_l, tool_r = st.columns([0.7, 0.3])
with tool_r:
    threshold = st.slider("Seuil de décision", 0.0, 1.0, float(DEF_THRESH), 0.01)

# ============================ Formulaire (groupé) ============================

with st.form("predict-form"):
    st.markdown("### Paramètres saisis")

    inputs: Dict[str, Any] = {}
    errors: List[str] = []

    def field(col: str, label: str, default_val: Any, help_txt: str = "", integer: bool = False):
        val_raw = st.text_input(label, value=str(default_val), help=help_txt, key=f"fld_{col}")
        val = parse_float_maybe(val_raw)
        if val is None:
            errors.append(f"« {label} » doit être un nombre (ex: 0,25).")
            return None
        if integer:
            val = float(int(round(val)))
        if col not in NO_BOUNDS and col in HARD_BOUNDS:
            lo, hi = HARD_BOUNDS[col]
            if lo is not None and val < lo: errors.append(f"« {label} » doit être ≥ {lo}.")
            if hi is not None and val > hi: errors.append(f"« {label} » doit être ≤ {hi}.")
        return val

    # Parcours par groupes, seulement pour les features présentes
    for group_name, cols in GROUPS.items():
        cols_present = [c for c in cols if c in feature_cols]
        if not cols_present:
            continue
        st.markdown(f"#### {group_name}")
        gc1, gc2, gc3 = st.columns(3)
        col_streams = [gc1, gc2, gc3]
        i = 0
        for col in cols_present:
            integer = (df is not None and col in df.columns and pd.api.types.is_integer_dtype(df[col])) or (col in LIKELY_INT_COLS)
            help_txt = ""
            if col in HARD_BOUNDS and col not in NO_BOUNDS:
                lo, hi = HARD_BOUNDS[col]
                mini = f"≥ {lo}" if lo is not None else ""
                maxi = f"≤ {hi}" if hi is not None else ""
                help_txt = f"Contrainte: {mini}{' / ' if mini and maxi else ''}{maxi}"
            label = col.replace("_", " ")
            with col_streams[i % 3]:
                inputs[col] = field(col, label, default_inputs.get(col, 0.0), help_txt, integer=integer)
                i += 1

    # Afficher les features non mappées dans GROUPS (si le modèle en a d'autres)
    others = [c for c in feature_cols if c not in sum(GROUPS.values(), [])]
    if others:
        st.markdown("#### Autres")
        oc1, oc2, oc3 = st.columns(3)
        ocols = [oc1, oc2, oc3]
        i = 0
        for col in others:
            integer = col in LIKELY_INT_COLS
            with ocols[i % 3]:
                inputs[col] = field(col, col.replace("_"," "), default_inputs.get(col, 0.0), integer=integer)
                i += 1

    # Aperçu tabulaire
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([inputs], columns=feature_cols), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔮 Prédire", use_container_width=True)

# ============================ Prédiction ============================

if submitted:
    if errors:
        st.error("Merci de corriger les champs suivants :")
        for e in errors:
            st.write(f"• {e}")
    else:
        X = pd.DataFrame([inputs], columns=feature_cols)
        if df is not None:
            for c in feature_cols:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    try:
                        X[c] = pd.to_numeric(pd.Series(X[c]), errors="coerce")
                    except Exception:
                        pass

        try:
            proba = float(predict_proba_safe(model, X)[0])
            pred = int(proba >= threshold)

            st.success("Prédiction effectuée ✅")

            cc1, cc2, cc3 = st.columns([0.3, 0.4, 0.3])
            with cc1:
                st.markdown("<div class='app-card'>", unsafe_allow_html=True)
                st.metric("Probabilité (classe 1)", f"{proba:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with cc2:
                st.markdown("<div class='app-card' style='display:flex;justify-content:center;'>", unsafe_allow_html=True)
                draw_gauge(proba)
                st.markdown("</div>", unsafe_allow_html=True)

            with cc3:
                st.markdown("<div class='app-card'>", unsafe_allow_html=True)
                pill = "<span class='pill pill-bad'>1 (défaut)</span>" if pred == 1 else "<span class='pill pill-ok'>0 (non défaut)</span>"
                st.markdown(f"**Décision**<br/>{pill}<br/><br/><span class='badge'>Seuil</span> {threshold:.2f}", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            result = {
                "inputs": inputs,
                "probability": proba,
                "decision": pred,
                "threshold": threshold,
            }
            st.session_state["history"].insert(0, result)
            st.session_state["history"] = st.session_state["history"][:10]

            export_json = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button("💾 Télécharger résultat (JSON)", data=export_json, file_name="prediction.json")

        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")

# ============================ Panneaux d'infos ============================

with st.expander("🧾 Détails & journal"):
    st.write(f"Chemin modèle : `{resolved_model_path}`")
    st.write(f"Variables détectées : {len(feature_cols)}")
    st.write("Sans seuils sur : " + (", ".join(sorted(NO_BOUNDS)) or "—"))
    if HARD_BOUNDS:
        st.write("Contraintes indispensables :", HARD_BOUNDS)

    if st.session_state["history"]:
        st.markdown("**Dernières prédictions**")
        st.table(pd.DataFrame(st.session_state["history"]))

with st.expander("💡 Astuces d’utilisation"):
    st.markdown(
        """
        - Tu peux **entrer « 0,06 »** ou « 0.06 » : les deux sont compris.
        - Les champs **FICO** et **lignes de crédit** sont **contrôlés** (domaine logique).
        - Le **seuil** s’ajuste en haut à droite ; il pilote la décision (0/1).
        - Le bouton **Télécharger** exporte la requête/réponse en JSON.
        """
    )
