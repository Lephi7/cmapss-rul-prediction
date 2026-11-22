# 	Charger les fichiers NASA et calculer les cibles RUL
#   soit créer les valeurs que le modèle doit apprendre 
#   à prédire, donc le nombre de cycles restants avant la panne.
#   RUL = dernier cycle de ce moteur - cycle actuel
#   Par exemple si un moteur tombe en panne au cycle 200:
#   cycle actuel = 50, le RUL a predire = 150

def load_data():
    # Charger les mesures d'entraînement et de test

def prepare_RUL(train):
    # Calculer les RUL d'entraînement et charger celles du test