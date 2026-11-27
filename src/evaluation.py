# Fonctions pour comparer les modeles
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(y_test, y_pred, name="model"):
    # evaluer les modeles
    # penalise les grosses erreurs
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    #erereur moyenne en cycles
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{name} : RMSE = {rmse:.2f}, MAE = {mae:.2f}")
    return rmse, mae