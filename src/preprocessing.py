# 	Préparer les variables : sélection des capteurs, 
#   imputation si nécessaire, standardisation.
#   ACP en option aussi

from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE       # Sélection de caractéristiques
from sklearn.preprocessing import StandardScaler                                # Mise à l'échelle
from sklearn.decomposition import PCA                                           # ACP (optionnel)
from sklearn.pipeline import make_pipeline


# pas besoin d'imputation, le dataset a pas de donnees manquantes

variance_seuil = 1e-4

def prepare_data(X_train, X_val, X_test):

def build_preprocessor():
    return make_pipeline(VarianceThreshold(variance_seuil), StandardScaler())