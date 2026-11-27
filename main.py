import matplotlib.pyplot as plt  # noqa: I001
import pandas as pd

from src.data import load_data, prepare_RUL
from src.evaluation import evaluate
from src.models.linear import build_linear_model, build_ridge_model
from src.plot_moteurs import plot_moteurs, plot_resultats, plot_predictions
from src.preprocessing import clip_rul

VARIANTES = [
    ("Référence",              build_linear_model, {},                            None),
    ("RUL plafonnée (125)",    build_linear_model, {},                            125), # 125 = moteur OK, 0 = Moteur dead
    ("MinMax",                 build_linear_model, {"scaler": "minmax"},          125),
    ("KBest (k=10)",           build_linear_model, {"k": 10},                     125),
    ("ACP 95 %",               build_linear_model, {"pca": 0.95},                 125),
    ("ACP 95 '%' + blanchiment", build_linear_model, {"pca": 0.95, "whiten": True}, 125),
    ("Ridge (alpha=10)",       build_ridge_model,  {"alpha": 10.0},               125),
]

def main():
    # data -> train : 100 moteurs, test : 100 moteurs 
    
    train, test = load_data() # mesures train / test
    y_train, y_test = prepare_RUL(train) # RUL à apprendre / vraies RUL du test

    # on garde tout sauf les id et les cycles pour entrainer le modele
    features = [c for c in train.columns if c not in ("engine_id", "cycle")]

    '''
    # visualise data des 3 premiers moteurs et du sensor2 a titre d'exemple
    plot_moteurs(train, test, enregistrer=True)
    plt.show()
    '''

    # Dernier cycle de chaque moteur de test
    test_last = test.groupby("engine_id").last() 

    # Entrainer les modeles de regression lineaire
    resultats_rl = []
    for nom, construire, options, plafond in VARIANTES:
        y_fit = clip_rul(y_train, plafond) if plafond else y_train  # plafond AVANT le fit

        model = construire(**options)
        model.fit(train[features], y_fit)

        y_pred = model.predict(test_last[features])  # RUL prédites
        rmse, mae = evaluate(y_test, y_pred, nom)
        resultats_rl.append({"Variante": nom, "RMSE": rmse, "MAE": mae})

    # Montre resultats de Regressions lineaires
    tableau = pd.DataFrame(resultats_rl).set_index("Variante").round(2)
    print("\n", tableau)

    # Plot les resultats
    plot_resultats(tableau, enregistrer=True)
    plt.show()

    # meilleure régression linéaire : RUL plafonnée
    best_RL = build_linear_model()
    best_RL.fit(train[features], clip_rul(y_train, 125))
    plot_predictions(y_test, best_RL.predict(test_last[features]), enregistrer=True)
    plt.show()



if __name__ == "__main__":
    main()