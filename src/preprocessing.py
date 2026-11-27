# 	Préparer les variables : sélection des capteurs, 
#   imputation si nécessaire, standardisation.
#   ACP en option aussi

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pas besoin d'imputation, le dataset a pas de donnees manquantes
VARIANCE_SEUIL = 1e-4
RUL_MAX = 150

def clip_rul(y, max_rul=RUL_MAX):
    # Plafonne la RUL (au besoin)
    return y.clip(upper=max_rul)

def build_preprocessor(scaler="standard", k=None, pca=None, whiten=False):
    etapes = [VarianceThreshold(VARIANCE_SEUIL)]                          # 12.2 : capteurs constants
    etapes.append(StandardScaler() if scaler == "standard" else MinMaxScaler())  # 12.1 : mise à l'échelle
    if k:
        etapes.append(SelectKBest(f_regression, k=k))                     # 12.2 : k meilleurs capteurs
    if pca:
        etapes.append(PCA(n_components=pca, whiten=whiten))               # 12.3 ACP, 12.7 blanchiment
    return make_pipeline(*etapes)