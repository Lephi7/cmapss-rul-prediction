"""Trace le nombre de cycles et un capteur des moteurs C-MAPSS FD001."""

from pathlib import Path

import matplotlib.pyplot as plt


CAPTEUR = "sensor_2"
MOTEURS = (1, 2, 3)
DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "figures"
    / "visualisation_FD001.png"
)


def plot_moteurs(train, test, *, enregistrer=False):
    fig = plt.figure(figsize=(12, 7.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.2))
    ax_train = fig.add_subplot(grid[0, 0])
    ax_test = fig.add_subplot(grid[0, 1], sharey=ax_train)
    ax_capteur = fig.add_subplot(grid[1, :])

    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.09)
    grid.update(wspace=0.19, hspace=0.62)
    fig.text(
        0.08, 0.945, "Cycles et mesures des moteurs",
        fontsize=22, fontweight="bold", color="#18243a",
    )
    fig.text(
        0.08, 0.903, "NASA C-MAPSS FD001 · Données simulées",
        fontsize=11, color="#4b5e72",
    )

    for ax, donnees, titre, couleur in (
        (ax_train, train, "Entraînement · jusqu’à la panne", "#286087"),
        (ax_test, test, "Test · observation interrompue avant la panne", "#00806e"),
    ):
        # Un cycle est compté une fois par moteur.
        cycles = donnees.groupby("engine_id")["cycle"].nunique()
        ax.bar(cycles.index, cycles.values, width=0.84, color=couleur)
        ax.set_title(titre, loc="left", fontweight="bold", fontsize=11, pad=10)
        ax.set_xlabel("Identifiant du moteur")
        ax.set_ylim(bottom=0)

    ax_train.set_ylabel("Nombre de cycles observés")
    ax_test.tick_params(labelleft=False)

    for moteur, couleur in zip(MOTEURS, ("#286087", "#a7511e", "#00806e")):
        mesure = train.loc[train["engine_id"] == moteur].sort_values("cycle")
        ax_capteur.plot(
            mesure["cycle"], mesure[CAPTEUR],
            label=f"Moteur {moteur}", color=couleur, linewidth=1.4,
        )

    ax_capteur.set_title(
        f"Évolution de {CAPTEUR} · moteurs d’entraînement",
        loc="left", fontweight="bold", fontsize=11, pad=10,
    )
    ax_capteur.set_xlabel("Cycle")
    ax_capteur.set_ylabel("Valeur mesurée")
    ax_capteur.legend(ncol=3, loc="upper left", frameon=False)

    for ax in (ax_train, ax_test, ax_capteur):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e0e7ef", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#9aabba")
        ax.tick_params(axis="both", labelsize=9, colors="#364960")

    if enregistrer:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(DESTINATION, dpi=160)
        print(f"Graphique enregistré dans {DESTINATION}")

    return fig


def main():
    from src.data import load_data

    train, test = load_data()
    fig = plot_moteurs(train, test, enregistrer=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
