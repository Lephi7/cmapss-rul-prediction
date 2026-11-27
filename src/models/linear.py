from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

from src.preprocessing import build_preprocessor


def build_linear_model(**options):
    # Regression lineaire
    return make_pipeline(build_preprocessor(**options), LinearRegression())

def build_ridge_model(alpha=1.0):
    # Régression linéaire régularisée, minimise erreur au carre et penalite sur taille des poids
    return make_pipeline(build_preprocessor(), Ridge(alpha=alpha)) # alpha = poids de la penalite