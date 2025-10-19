import joblib
import pandas as pd
import warnings
from pprint import pprint  # Pour un affichage plus joli
from sklearn.pipeline import Pipeline
# --- 1. CONFIGURATION ---
MODEL_FILE = "model.pkl"  

# On ignore les avertissements si ton scikit-learn local
# n'a pas la même version que celui qui a créé le modèle
warnings.filterwarnings("ignore", category=UserWarning)

# --- 2. CHARGEMENT ---
print(f"Chargement du modèle depuis : {MODEL_FILE}...")
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Modèle chargé avec succès !")
except FileNotFoundError:
    print(f"❌ ERREUR : Fichier '{MODEL_FILE}' introuvable.")
    print("Vérifie qu'il est bien dans le même dossier que ce script.")
    exit()  # Arrête le script si le fichier n'est pas là
except Exception as e:
    print(f"❌ ERREUR : Impossible de charger le modèle. {e}")
    exit()

# --- 3. INSPECTION (PARAMÈTRES) ---
print("\n" + "="*40)
print("⚙️ 1. PARAMÈTRES (Les réglages du modèle)")
print("="*40)
pprint(model.get_params())

# --- 4. INSPECTION (COEFFICIENTS) ---
print("\n" + "="*40)
print("🧠 2. COEFFICIENTS (L'impact de chaque variable)")
print("="*40)

# On vérifie si c'est bien un Pipeline
if not isinstance(model, Pipeline):
    print("Ce script est fait pour un Pipeline Scikit-Learn.")
    print("Le modèle chargé n'est pas un Pipeline. Arrêt.")

else:
    try:
        # Étape 1: Accéder au classifieur (nommé 'clf' dans ton Pipeline)
        estimator = model.named_steps['clf']
        
        # Étape 2: Accéder au préprocesseur (nommé 'pre' dans ton Pipeline)
        preprocessor = model.named_steps['pre']

        # Étape 3: Vérifier qu'on a bien les coefficients ET les noms
        if hasattr(estimator, "coef_") and hasattr(preprocessor, "get_feature_names_out"):
            
            # Récupère les noms des features (après le StandardScaler)
            # ex: 'num__fico_score', 'num__income', etc.
            feature_names = preprocessor.get_feature_names_out()
            
            # Récupère les coefficients (poids)
            coefficients = estimator.coef_[0]

            # Crée le DataFrame
            df_impact = pd.DataFrame(
                coefficients,
                index=feature_names,
                columns=["Impact sur le risque de défaut (logit)"]
            )
            
            print(df_impact.sort_values(by="Impact sur le risque de défaut (logit)", ascending=False))
        
        else:
            print("Impossible de trouver '.coef_' sur l'étape 'clf' ou '.get_feature_names_out' sur l'étape 'pre'.")

    except KeyError:
        print("ERREUR: Le Pipeline n'a pas les étapes nommées 'pre' et 'clf'.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")