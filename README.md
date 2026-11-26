# cmapss-rul-prediction
Prédiction de la durée de vie restante de moteurs d’avion sur NASA C-MAPSS : comparaison entre régression linéaire, SVR et perceptron multicouche.


## Installation

Depuis la racine du projet, crée et active un environnement Python,
puis installe les dépendances listées dans [requirements.txt](requirements.txt).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## main.py

## Regression lineaire 
1. load data
2. pretraitement -> standardiser
3. retirer capteurs constant
4. ACP
