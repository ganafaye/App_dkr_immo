#!/bin/bash
# Nettoyage des anciens processus
pkill -f uvicorn
pkill -f streamlit

# 1. Lancement de l'API FastAPI (Port 8000) en arrière-plan
uvicorn main:app --reload --port 8000 &

# 2. Lancement du Dashboard Streamlit (Port 8501) en arrière-plan
# --server.headless true empêche l'ouverture auto du navigateur
streamlit run admin_dash.py --server.port 8501 --server.headless true &

echo "🚀 Systèmes démarrés !"
echo "👉 App Client : http://127.0.0.1:8000"
echo "👉 Accès Admin : http://127.0.0.1:8000/admin"