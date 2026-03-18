import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import MODEL_PATH, MODEL_METRICS


def run_retraining(data_path):
    # 1. Chargement des nouvelles données (ex: CSV scrapé récemment)
    df = pd.read_csv(data_path)

    # 2. Prétraitement & Entraînement
    # (Ici tu reprends ta logique de ton Notebook)
    X = df.drop(columns=['price_log'])
    y = df['price_log']

    new_model = RandomForestRegressor(n_estimators=100)
    new_model.fit(X, y)
    new_score = new_model.score(X, y)  # Simplifié pour l'exemple

    # 3. Logique MLOps : Le "Gatekeeper"
    if new_score > MODEL_METRICS["R2_SCORE"]:
        print(f"🚀 Nouveau Champion détecté ! Score: {new_score:.2f} > {MODEL_METRICS['R2_SCORE']}")

        # Sauvegarde du nouveau pack d'assets
        assets = {
            'model': new_model,
            'quartier_map': {...},  # À extraire de tes données
            'features': list(X.columns)
        }
        joblib.dump(assets, MODEL_PATH)
        return True
    else:
        print("❌ Le modèle Challenger n'est pas meilleur que le Champion actuel.")
        return False