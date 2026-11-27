"""Trace le nombre de cycles et un capteur des moteurs C-MAPSS FD001."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CAPTEUR = "sensor_2"
MOTEURS = (1, 2, 3)
FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
DESTINATION_EXEMPLE = FIG_DIR / "visualisation_FD001.png"
DESTINATION_LINEAR = FIG_DIR / "resultats_regression_lineaire.png"


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
        DESTINATION_EXEMPLE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(DESTINATION_EXEMPLE, dpi=160)
        print(f"Graphique enregistré dans {DESTINATION_EXEMPLE}")

    return fig

def plot_resultats(tableau, titre="Régression linéaire", *, enregistrer=False):
    x = np.arange(len(tableau))  # une position par variante
    largeur = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.25)
    fig.text(
        0.08, 0.92, f"{titre} · comparaison des variantes",
        fontsize=18, fontweight="bold", color="#18243a",
    )
    fig.text(
        0.08, 0.87, "NASA C-MAPSS FD001 · Erreur sur les 100 moteurs de test (en cycles)",
        fontsize=11, color="#4b5e72",
    )

    b1 = ax.bar(x - largeur / 2, tableau["RMSE"], largeur, label="RMSE", color="#286087")
    b2 = ax.bar(x + largeur / 2, tableau["MAE"], largeur, label="MAE", color="#00806e")
    ax.bar_label(b1, fmt="%.1f", fontsize=8, color="#364960", padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, color="#364960", padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(tableau.index, rotation=25, ha="right")
    ax.set_ylabel("Erreur (cycles)")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, loc="upper right", frameon=False)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e7ef", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#9aabba")
    ax.tick_params(axis="both", labelsize=9, colors="#364960")

    rmse_min = tableau["RMSE"].min()
    ax.axhline(rmse_min, ls="--", lw=1, color="#9aabba", zorder=0)
    ax.text(len(tableau) - 0.5, rmse_min, f"RMSE min = {rmse_min:.1f}",
            ha="right", va="bottom", fontsize=8, color="#530000")

    if enregistrer:
        DESTINATION_LINEAR.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(DESTINATION_LINEAR, dpi=160)
        print(f"Graphique enregistré dans {DESTINATION_LINEAR}")

    return fig