import numpy as np
import pandas as pd


def preprocess_input(data, quartier_map, features_names):
    """
    Transforme les entrées brutes du formulaire en vecteur compatible avec le modèle.

    Args:
        data (dict): Dictionnaire contenant les entrées (quartier, surface, etc.)
        quartier_map (dict): Dictionnaire de Target Encoding (Quartier -> Prix Moyen)
        features_names (list): Liste ordonnée des colonnes attendues par le modèle

    Returns:
        tuple: (DataFrame pour le modèle, score de luxe entier)
    """

    # 1. Target Encoding du quartier
    # Si le quartier n'est pas dans le dictionnaire (ex: nouveau quartier),
    # on utilise la médiane pour rester robuste.
    q_score = quartier_map.get(data['quartier'], np.median(list(quartier_map.values())))

    # 2. Features Engineering (identique à ton code Streamlit d'origine)
    # Calcul du score de luxe (0 à 3)
    luxe = int(data['meuble']) + int(data['neuf']) + int(data['vue_mer'])

    # Interaction entre surface et standing
    surf_standing = data['surface'] * luxe

    # Ratio d'équipement par chambre
    ratio = data['sdb'] / max(data['chambres'], 1)

    # 3. Création du DataFrame final
    # L'ordre des colonnes doit être STRICTEMENT identique à celui de l'entraînement
    input_df = pd.DataFrame([[
        data['surface'],  # Feature 1
        data['sdb'],  # Feature 2
        q_score,  # Feature 3 (Target Encoded)
        int(data['meuble']),  # Feature 4
        int(data['neuf']),  # Feature 5
        int(data['vue_mer']),  # Feature 6
        ratio,  # Feature 7 (Calculée)
        surf_standing  # Feature 8 (Calculée)
    ]], columns=features_names)

    return input_df, luxe