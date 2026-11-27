# 	Charger les fichiers NASA et calculer les cibles RUL
#   soit créer les valeurs que le modèle doit apprendre 
#   à prédire, donc le nombre de cycles restants avant la panne.
#   RUL = dernier cycle de ce moteur - cycle actuel
#   Par exemple si un moteur tombe en panne au cycle 200:
#   cycle actuel = 50, le RUL a predire = 150

from pathlib import Path
import pandas as pd # pandas plus pratique pour accéder aux colonnes par leur nom et regrouper les lignes par moteur


# Chemin depuis src/data.py vers data/raw/FD001/
DATA_DIR = (
    Path(__file__).resolve().parent.parent  # va chercher le path du folder du haut
    / "data" / "raw" / "FD001"
)

# Noms des 26 colonnes des fichiers NASA
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data():
    # Charger les mesures d'entraînement et de test
    train = pd.read_csv(
        DATA_DIR / "train_FD001.txt",
        sep=r"\s+",                         # permet de lire les valeurs séparées par plusieurs espaces
        header=None,
        names=COLUMNS
    )

    test = pd.read_csv(
        DATA_DIR / "test_FD001.txt",
        sep=r"\s+",                         # permet de lire les valeurs séparées par plusieurs espaces
        header=None,
        names=COLUMNS
    )

    return train, test


def prepare_RUL(train):
    # Calculer les RUL d'entraînement et charger celles du test
    # Dernier cycle de chaque moteur, repeter sur chaque ligne (pour faire Last - actuel)
    last_cycle = (
        train.groupby("engine_id")["cycle"].transform("max") # groupby(...).transform("max") conserve une valeur max par ligne de #moteur pour effectuer la soustraction
    )

    # Une cible pour chaque ligne d'entraînement
    y_train = (last_cycle - train["cycle"]).rename("RUL")

    # Une cible par moteur de test
    y_test = pd.read_csv(
        DATA_DIR / "RUL_FD001.txt",
        sep=r"\s+",             
        header=None,
        names=["RUL"]
    )

    # Dans le fichier NASA, les RUL suivent les moteurs 1, 2, 3...
    y_test.index = pd.Index(
        range(1, len(y_test) + 1),
        name="engine_id"
    )

    return y_train, y_test