import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from src.data import load_data
from src.plot_moteurs import plot_moteurs
from src.preprocessing import build_preprocessor


def main():
    # data -> train : 100 moteurs, test : 100 moteurs 
    train, test = load_data()
    print(train.shape, test.shape)

    # visualise data des 3 premiers moteurs et du sensor2 a titre d'exemple
    plot_moteurs(train, test, enregistrer=True)
    plt.show()


    model = make_pipeline(build_preprocessor(), LinearRegression())
    model.fit(train, y_train)
    y_pred = model.predict(test)




if __name__ == "__main__":
    main()