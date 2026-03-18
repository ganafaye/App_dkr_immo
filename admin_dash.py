import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import shutil
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import json
from dotenv import load_dotenv
import psutil
import platform

load_dotenv()

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Dakar Immo AI - Admin MLOps",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #e5e7eb;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    .status-success {
        background: #10b981;
        color: white;
    }

    .status-warning {
        background: #f59e0b;
        color: white;
    }

    .status-error {
        background: #ef4444;
        color: white;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }

    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .promote-btn > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
    }

    .reject-btn > button {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        border: none;
    }

    .archive-card {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    .progress-bar {
        width: 100%;
        height: 0.5rem;
        background: #e5e7eb;
        border-radius: 9999px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 9999px;
        transition: width 0.3s ease;
    }

    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6b7280;
        font-size: 0.875rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- SYSTÈME DE LOGIN AMÉLIORÉ ---
def check_password():
    """Retourne True si l'utilisateur est authentifié."""

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        # Interface de login stylisée
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 3rem 0;">
                    <span style="font-size: 4rem;">🏗️</span>
                    <h1 style="color: #1f2937; margin: 1rem 0;">Dakar Immo AI</h1>
                    <p style="color: #6b7280; margin-bottom: 2rem;">Centre de Contrôle MLOps - Accès Administrateur</p>
                </div>
            """, unsafe_allow_html=True)

            with st.form("login_form"):
                password = st.text_input("Mot de passe administrateur", type="password",
                                         placeholder="Entrez votre mot de passe")
                submit = st.form_submit_button("Se connecter", use_container_width=True)

                if submit:
                    if password == os.getenv("ADMIN_PASSWORD", "admin123"):
                        st.session_state["password_correct"] = True
                        st.session_state["username"] = "Admin"
                        st.rerun()
                    else:
                        st.error("❌ Mot de passe incorrect")
            return False
    return True


# --- VÉRIFICATION AUTHENTIFICATION ---
if not check_password():
    st.stop()

# --- INITIALISATION DES CHEMINS ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'modele_immo_dakar.pkl')
MAP_PATH = os.path.join(MODEL_DIR, 'quartier_mapping.joblib')
ARCHIVE_DIR = os.path.join(MODEL_DIR, 'archive')
LOG_DIR = 'logs'

# Création des dossiers
for dir_path in [MODEL_DIR, ARCHIVE_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# --- FONCTIONS UTILITAIRES ---
def get_model_info():
    """Récupère les informations du modèle actuel."""
    info = {
        'exists': os.path.exists(MODEL_PATH),
        'path': MODEL_PATH,
        'size': None,
        'modified': None,
        'version': None,
        'metrics': {}
    }

    if info['exists']:
        info['size'] = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        info['modified'] = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))

        try:
            model_data = joblib.load(MODEL_PATH)
            info['version'] = model_data.get('version', '1.0.0')
            info['metrics'] = model_data.get('metrics', {})
        except:
            pass

    return info


def get_archive_list():
    """Liste les archives disponibles."""
    archives = []
    for f in os.listdir(ARCHIVE_DIR):
        if f.startswith('model_') and f.endswith('.pkl'):
            archive_path = os.path.join(ARCHIVE_DIR, f)
            map_file = f.replace('model_', 'map_').replace('.pkl', '.joblib')
            map_path = os.path.join(ARCHIVE_DIR, map_file)

            if os.path.exists(map_path):
                archives.append({
                    'model': f,
                    'map': map_file,
                    'path': archive_path,
                    'date': datetime.fromtimestamp(os.path.getmtime(archive_path)),
                    'size': os.path.getsize(archive_path) / (1024 * 1024)
                })

    return sorted(archives, key=lambda x: x['date'], reverse=True)


def get_system_health():
    """Récupère les métriques système."""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'uptime': (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600
    }


# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 3rem;">🏗️</span>
            <div>
                <h1 style="margin: 0; font-size: 2rem;">Dakar Immo AI</h1>
                <p style="margin: 0; opacity: 0.9;">Centre de Contrôle MLOps - Administration du Modèle Random Forest</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR AMÉLIORÉE ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🏗️</span>
            <h3 style="color: #1f2937; margin: 0.5rem 0;">Dakar Immo AI</h3>
            <p style="color: #6b7280; font-size: 0.875rem;">Centre de Contrôle MLOps</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # État du système
    st.subheader("🖥️ État du système")
    health = get_system_health()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU", f"{health['cpu_percent']:.0f}%")
    with col2:
        st.metric("RAM", f"{health['memory_percent']:.0f}%")

    st.progress(health['cpu_percent'] / 100, text="CPU")
    st.progress(health['memory_percent'] / 100, text="Mémoire")
    st.progress(health['disk_percent'] / 100, text="Disque")

    st.divider()

    # Informations du modèle
    st.subheader("📊 Modèle actuel")
    model_info = get_model_info()

    if model_info['exists']:
        st.markdown(f"""
            <div style="background: #f0fdf4; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #10b981;">
                <p style="color: #059669; font-weight: 600; margin: 0;">✅ Modèle actif</p>
                <p style="color: #1f2937; font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                    Version: {model_info['version']}<br>
                    Taille: {model_info['size']:.2f} MB<br>
                    MAJ: {model_info['modified'].strftime('%d/%m/%Y %H:%M')}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: #fef2f2; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ef4444;">
                <p style="color: #dc2626; font-weight: 600; margin: 0;">❌ Aucun modèle</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Utilisateur connecté
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0;">
            <span style="background: #667eea; color: white; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-weight: bold;">A</span>
            <div>
                <p style="margin: 0; font-weight: 600;">Administrateur</p>
                <p style="margin: 0; font-size: 0.75rem; color: #6b7280;">Connecté</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Déconnexion", use_container_width=True):
        st.session_state["password_correct"] = False
        st.rerun()

# --- DASHBOARD PRINCIPAL ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard MLOps", "⚙️ Entraînement", "📦 Archives & Rollback"])

# --- TAB 1: DASHBOARD MLOPS ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">📈 Score R²</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{model_info["metrics"].get("r2", 0.55):.1%}</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #10b981; font-size: 0.875rem;">▲ +0.02 depuis dernier</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">💰 MAE</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{model_info["metrics"].get("mae", 45000):,.0f} FCFA</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color: #10b981; font-size: 0.875rem;">▼ -5% vs précédent</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">📊 Prédictions</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-value">15,234</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #6b7280; font-size: 0.875rem;">Dernières 24h: 342</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">⏱️ Latence</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-value">124 ms</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #10b981; font-size: 0.875rem;">p95: 187 ms</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Graphiques de performance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Évolution du score R²")
        # Données simulées
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        r2_scores = 0.55 + np.cumsum(np.random.normal(0.005, 0.01, 30))
        r2_scores = np.clip(r2_scores, 0.5, 0.65)

        fig = px.line(x=dates, y=r2_scores, labels={'x': 'Date', 'y': 'Score R²'})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Distribution des erreurs")
        # Données simulées
        errors = np.random.normal(45000, 15000, 1000)

        fig = px.histogram(errors, nbins=30, labels={'value': 'Erreur (FCFA)'})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Importance des features
    st.subheader("🌟 Importance des features")

    features = ['Surface', 'Quartier', 'Chambres', 'SDB', 'Meublé', 'Neuf', 'Vue Mer', 'Ratio SDB/Chambre']
    importances = [35, 25, 15, 10, 5, 4, 3, 3]

    fig = px.bar(x=importances, y=features, orientation='h', labels={'x': 'Importance (%)', 'y': ''})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ENTRAÎNEMENT ---
with tab2:
    st.subheader("📤 Import des nouvelles données")

    uploaded_file = st.file_uploader(
        "Charger le fichier CSV des nouvelles annonces",
        type="csv",
        help="Format attendu: quartier, surface_m2, chambres, salles_de_bain, meuble, neuf, vue_mer, prix"
    )

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

        # KPIs de santé des données
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">📊 Annonces</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{len(df_raw):,}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            missing_pct = (df_raw.isnull().sum().sum() / np.prod(df_raw.shape)) * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">❌ Manquantes</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{missing_pct:.1f}%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            outliers = len(df_raw[df_raw['prix'] > 2000000]) if 'prix' in df_raw.columns else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">⚠️ Hors normes</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{outliers}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            unique_quartiers = df_raw['quartier'].nunique() if 'quartier' in df_raw.columns else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">📍 Quartiers</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{unique_quartiers}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Visualisation des données
        with st.expander("📊 Voir l'analyse exploratoire"):
            tab_dist, tab_corr = st.tabs(["Distribution des prix", "Corrélations"])

            with tab_dist:
                if 'prix' in df_raw.columns:
                    df_filtered = df_raw[df_raw['prix'] <= 2000000]
                    fig = px.histogram(df_filtered, x='prix', nbins=50,
                                       title="Distribution des prix (≤ 2M FCFA)")
                    st.plotly_chart(fig, use_container_width=True)

            with tab_corr:
                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df_raw[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🚀 Entraînement du Challenger")

        # Paramètres d'entraînement
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Nombre d'arbres", 100, 1000, 500, 50)
            max_depth = st.slider("Profondeur max", 5, 30, 15)
        with col2:
            min_samples_split = st.slider("Min samples split", 2, 20, 5)
            min_samples_leaf = st.slider("Min samples leaf", 1, 10, 2)

        if st.button("🏋️ Lancer l'entraînement", use_container_width=True, type="primary"):
            with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes."):
                try:
                    # Simulation de progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 1. Nettoyage
                    status_text.text("🧹 Nettoyage des données...")
                    df_clean = df_raw[df_raw['prix'] <= 2000000].copy()
                    progress_bar.progress(20)

                    # 2. Target Encoding
                    status_text.text("📍 Encodage des quartiers...")
                    quartier_map = df_clean.groupby('quartier')['prix'].median().to_dict()
                    df_clean['quartier_score'] = df_clean['quartier'].map(quartier_map)
                    progress_bar.progress(40)

                    # 3. Feature engineering
                    status_text.text("🔧 Création des features...")
                    df_clean['ratio_sdb_chambre'] = df_clean['salles_de_bain'] / df_clean['chambres'].replace(0, 1)
                    df_clean['standing_score'] = df_clean['meuble'] + df_clean['neuf'] + df_clean['vue_mer']
                    df_clean['surface_standing'] = df_clean['surface_m2'] * df_clean['standing_score']
                    progress_bar.progress(60)

                    # 4. Préparation des features
                    status_text.text("📊 Préparation des données...")
                    feature_cols = ['surface_m2', 'salles_de_bain', 'quartier_score', 'meuble',
                                    'neuf', 'vue_mer', 'ratio_sdb_chambre', 'surface_standing']

                    X = df_clean[feature_cols].fillna(0)
                    y = np.log1p(df_clean['prix'])  # Log transformation

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    progress_bar.progress(80)

                    # 5. Entraînement
                    status_text.text("🌲 Entraînement du modèle Random Forest...")
                    challenger = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        n_jobs=-1,
                        random_state=42
                    )
                    challenger.fit(X_train, y_train)

                    # 6. Évaluation
                    status_text.text("📈 Évaluation du modèle...")
                    y_pred_log = challenger.predict(X_test)
                    y_pred = np.expm1(y_pred_log)
                    y_true = np.expm1(y_test)

                    # Métriques
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                    # Cross-validation
                    cv_scores = cross_val_score(challenger, X_train, y_train, cv=5, scoring='r2')

                    progress_bar.progress(100)
                    status_text.text("✅ Entraînement terminé !")
                    time.sleep(0.5)

                    # Résultats
                    st.success("🎯 Modèle challenger entraîné avec succès !")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Score R²", f"{r2:.3f}", f"{r2 - model_info['metrics'].get('r2', 0.55):+.3f}")
                    col2.metric("MAE", f"{mae:,.0f} FCFA", f"{model_info['metrics'].get('mae', 45000) - mae:+,.0f}")
                    col3.metric("RMSE", f"{rmse:,.0f} FCFA")

                    # Comparaison avec champion
                    st.subheader("⚖️ Comparaison Champion vs Challenger")

                    comparison_data = {
                        'Métrique': ['R² Score', 'MAE (FCFA)', 'RMSE (FCFA)'],
                        'Champion': [model_info['metrics'].get('r2', 0.55),
                                     model_info['metrics'].get('mae', 45000),
                                     model_info['metrics'].get('rmse', 65000)],
                        'Challenger': [r2, mae, rmse]
                    }

                    df_comp = pd.DataFrame(comparison_data)
                    st.dataframe(df_comp, use_container_width=True)

                    # Boutons de déploiement
                    st.subheader("📦 Déploiement")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("✅ Promouvoir", use_container_width=True, key="promote"):
                            # Archiver l'ancien modèle
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            if os.path.exists(MODEL_PATH):
                                shutil.copy(MODEL_PATH, f"{ARCHIVE_DIR}/model_{timestamp}.pkl")
                                shutil.copy(MAP_PATH, f"{ARCHIVE_DIR}/map_{timestamp}.joblib")

                            # Sauvegarder le nouveau modèle
                            model_data = {
                                'model': challenger,
                                'features': feature_cols,
                                'quartier_map': quartier_map,
                                'version': f"2.0.{timestamp}",
                                'metrics': {
                                    'r2': r2,
                                    'mae': mae,
                                    'rmse': rmse,
                                    'cv_mean': cv_scores.mean(),
                                    'cv_std': cv_scores.std()
                                },
                                'timestamp': datetime.now().isoformat()
                            }

                            joblib.dump(model_data, MODEL_PATH)
                            joblib.dump(quartier_map, MAP_PATH)

                            st.balloons()
                            st.success("🚀 Le challenger est maintenant le champion !")
                            time.sleep(1)
                            st.rerun()

                    with col2:
                        if st.button("❌ Rejeter", use_container_width=True, key="reject"):
                            st.warning("Challenger rejeté")

                    with col3:
                        if st.button("💾 Sauvegarder", use_container_width=True, key="save"):
                            # Sauvegarder comme challenger sans déployer
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            challenger_path = os.path.join(ARCHIVE_DIR, f"challenger_{timestamp}.pkl")

                            model_data = {
                                'model': challenger,
                                'features': feature_cols,
                                'quartier_map': quartier_map,
                                'metrics': {
                                    'r2': r2,
                                    'mae': mae,
                                    'rmse': rmse
                                },
                                'timestamp': datetime.now().isoformat()
                            }

                            joblib.dump(model_data, challenger_path)
                            st.success(f"✅ Challenger sauvegardé dans les archives")

                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")

# --- TAB 3: ARCHIVES & ROLLBACK ---
with tab3:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📦 Archives disponibles")

        archives = get_archive_list()

        if archives:
            for archive in archives[:10]:  # Afficher les 10 plus récentes
                with st.container():
                    st.markdown(f"""
                        <div class="archive-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <p style="font-weight: 600; margin: 0;">Version {archive['date'].strftime('%Y%m%d_%H%M')}</p>
                                    <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">
                                        {archive['date'].strftime('%d/%m/%Y %H:%M')} • {archive['size']:.2f} MB
                                    </p>
                                </div>
                                <span class="status-badge status-success">Archive</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
                    with col_btn1:
                        if st.button(f"🔍 Voir", key=f"view_{archive['model']}"):
                            st.info(f"Détails de l'archive {archive['model']}")
                    with col_btn2:
                        if st.button(f"↩️ Restaurer", key=f"restore_{archive['model']}"):
                            try:
                                # Restaurer le modèle
                                shutil.copy(archive['path'], MODEL_PATH)
                                # Restaurer le mapping
                                map_archive = os.path.join(ARCHIVE_DIR, archive['map'])
                                shutil.copy(map_archive, MAP_PATH)
                                st.success("✅ Modèle restauré avec succès !")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erreur: {e}")
        else:
            st.info("📭 Aucune archive disponible")

    with col2:
        st.subheader("🔄 Rollback")

        st.markdown("""
            <div style="background: #fef3c7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
                <p style="color: #92400e; font-weight: 600; margin: 0;">⚠️ Attention</p>
                <p style="color: #92400e; font-size: 0.875rem; margin: 0.5rem 0 0 0;">
                    La restauration d'une archive remplacera le modèle actuel. Assurez-vous d'avoir sauvegardé les changements récents.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Statistiques des archives
        archives = get_archive_list()
        if archives:
            st.metric("Nombre d'archives", len(archives))
            st.metric("Espace total", f"{sum(a['size'] for a in archives):.2f} MB")
            st.metric("Dernière archive", archives[0]['date'].strftime('%d/%m/%Y'))

        # Bouton de nettoyage
        if st.button("🧹 Nettoyer les anciennes archives (> 30 jours)", use_container_width=True):
            cutoff = datetime.now() - pd.Timedelta(days=30)
            deleted = 0

            for archive in archives:
                if archive['date'] < cutoff:
                    try:
                        os.remove(archive['path'])
                        map_path = os.path.join(ARCHIVE_DIR, archive['map'])
                        if os.path.exists(map_path):
                            os.remove(map_path)
                        deleted += 1
                    except:
                        pass

            st.success(f"✅ {deleted} archives supprimées")

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        <p>Dakar Immo AI • Centre de Contrôle MLOps v2.0</p>
        <p style="font-size: 0.75rem;">© 2024 - Tous droits réservés</p>
    </div>
""", unsafe_allow_html=True)