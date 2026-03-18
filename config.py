from pathlib import Path
import numpy as np

# --- CHEMINS DE BASE ---
# Localise le dossier racine du projet (où se trouve main.py)
BASE_DIR = Path(__file__).resolve().parent

# Chemins vers les dossiers clés
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# --- CONFIGURATION DU MODÈLE ---
# Nom du fichier du modèle champion
MODEL_FILENAME = "modele_immo_dakar.pkl"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# --- MÉTRIQUES DE PERFORMANCE (Issues de ton rapport technique) ---
# Ces valeurs peuvent être affichées sur ton dashboard [cite: 121, 149]
MODEL_METRICS = {
    "R2_SCORE": 0.55,
    "MAE_TARGET": 5.0, # Seuil pour le mécanisme Challenger [cite: 121]
    "LAST_UPDATE": "Mars 2026"
}

# --- PARAMÈTRES PAR DÉFAUT POUR L'INFÉRENCE ---
# Utilisés par ton preprocessor en cas de données manquantes [cite: 81]
DEFAULT_SETTINGS = {
    "MIN_SURFACE": 15,
    "MAX_SURFACE": 1000,
    "CURRENCY": "FCFA"
}

# --- CONFIGURATION SERVEUR ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
DEBUG_MODE = True