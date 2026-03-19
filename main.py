import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import uvicorn

# Imports de tes modules locaux
from config import MODEL_PATH, TEMPLATES_DIR, SERVER_HOST, SERVER_PORT
from api.preprocessor import preprocess_input

app = FastAPI(title="Dakar Rent Predictor")
# 1. Monter le dossier static pour les icônes (déjà fait ?)
app.mount("/static", StaticFiles(directory="static"), name="static")


# 2. Les routes PWA spécifiques pour la racine
# Ces fichiers sont physiquement dans /static/ mais servis à la racine.

@app.get("/manifest.json")
async def get_manifest():
    return FileResponse("static/manifest.json", media_type="application/json")

@app.get("/sw.js")
async def get_sw():
    # Attention, le type mime 'application/javascript' est important !
    return FileResponse("static/sw.js", media_type="application/javascript")
# Configuration des templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- CHARGEMENT DU MODÈLE CHAMPION --- [cite: 131, 197]
try:
    assets = joblib.load(str(MODEL_PATH))
    model = assets['model']
    quartier_map = assets['quartier_map']
    features_names = assets['features']
    print(f"✅ Pipeline MLOps : Modèle Champion chargé ({len(quartier_map)} quartiers)")
except Exception as e:
    print(f"❌ Erreur critique lors du chargement des assets : {e}")
    model = quartier_map = features_names = None

# --- ROUTES DE NAVIGATION ---

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Page d'accueil"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def read_prediction(request: Request):
    """Page de prédiction standard"""
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "quartiers": sorted(quartier_map.keys()) if quartier_map else []
    })

@app.get("/prediction", response_class=HTMLResponse)
async def read_prediction_v2(request: Request):
    """Page de prédiction version avancée"""
    return templates.TemplateResponse("teste.html", {
        "request": request,
        "quartiers": sorted(quartier_map.keys()) if quartier_map else []
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Visualisation des données"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    """Documentation et Méthodologie"""
    return templates.TemplateResponse("about.html", {"request": request})

# --- ROUTE D'INFÉRENCE (LOGIQUE MÉTIER) --- [cite: 140]

@app.post("/predict")
async def predict(
    request: Request,
    quartier: str = Form(...),
    surface: float = Form(...),
    chambres: int = Form(...),
    sdb: int = Form(...),
    meuble: bool = Form(False),
    neuf: bool = Form(False),
    vue_mer: bool = Form(False)
):
    if not model:
        return {"error": "Modèle non disponible"}

    # Regroupement des données pour le préprocesseur
    raw_data = {
        "quartier": quartier, "surface": surface, "chambres": chambres,
        "sdb": sdb, "meuble": meuble, "neuf": neuf, "vue_mer": vue_mer
    }

    # 1. Prétraitement via le module api/
    input_df, luxe_score = preprocess_input(raw_data, quartier_map, features_names)

    # 2. Inférence (conversion Log -> Prix réel via np.expm1)
    prediction_log = model.predict(input_df)
    prix_final = np.expm1(prediction_log)[0]

    # 3. Formatage pour l'affichage utilisateur [cite: 145]
    prix_formate = f"{round(prix_final, -3):,}".replace(",", " ")

    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "resultat": prix_formate,
        "quartier": quartier,
        "score_luxe": luxe_score,
        "quartiers": sorted(quartier_map.keys()) # Pour garder la liste dans le select
    })

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "n_quartiers": len(quartier_map) if quartier_map else 0
    }
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse
load_dotenv()
security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("ADMIN_USERNAME")
    correct_password = os.getenv("ADMIN_PASSWORD") # Ton mot de passe sécurisé
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/admin")
async def guest_to_admin(username: str = Depends(get_current_username)):
    # Seuls les utilisateurs connectés atteignent cette ligne
    return RedirectResponse(url="http://127.0.0.1:8501")
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 DAKAR RENT PREDICTOR - DÉPLOIEMENT")
    print("="*50)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, reload=True)