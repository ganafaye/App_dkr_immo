# 🏠 Dakar Immo AI Predictor v2.0
> **Système intelligent de prédiction et d'analyse du marché immobilier à Dakar.**

[![Python CI/CD Pipeline](https://github.com/ganafaye/App_dkr_immo/actions/workflows/main.yml/badge.svg)](https://github.com/ganafaye/App_dkr_immo/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

## 📝 Présentation du Projet
Développé dans le cadre du module de **Machine Learning Master 1 SI **, ce projet vise à fournir une estimation précise des prix de location à Dakar en utilisant des algorithmes d'apprentissage supervisé.

L'application intègre un pipeline complet : du nettoyage des données (Scraping/Cleaning) au déploiement continu (CI/CD).

## 🚀 Fonctionnalités Clés
- **Prédiction en temps réel** : Estimation basée sur la localité (Almadies, Plateau, Maristes, etc.), la surface et le type de bien.
- **Dashboard Admin** : Interface Streamlit pour visualiser les métriques du modèle et explorer les données.
- **API REST** : Endpoint FastAPI documenté pour une intégration tierce.
- **Pipeline MLOps** : Déploiement automatisé sur **Render** via GitHub Actions.

## 📊 Performance du Modèle
Le modèle actuel utilise l'algorithme **Random Forest / Extra Trees** :
- **Précision ($R^2$)** : ~85%
- **Données** : Plus de 47 quartiers de Dakar cartographiés.
- **Gestion des Outliers** : Traitement spécifique pour les biens de luxe (> 2M FCFA).

## 🛠️ Stack Technique
- **Langage** : Python 3.10+
- **Bibliothèques** : Pandas, Scikit-Learn, Plotly, Jinja2.
- **Backend** : FastAPI & Uvicorn.
- **Frontend** : Streamlit & HTML/CSS (Jinja Templates).
- **DevOps** : GitHub Actions, Git, Render.

## ⚙️ Installation & Utilisation locale
1. **Cloner le projet** :
   ```bash
   git clone [https://github.com/ganafaye/App_dkr_immo.git](https://github.com/ganafaye/App_dkr_immo.git)
   cd App_dkr_immo
