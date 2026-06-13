# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 10:53:41 2025

@author: frederic.ros
"""

import matplotlib.pyplot as plt
def plot_accuracy(history, title = "Accuracy"):
    """
    Affiche l'évolution de l'accuracy au fil des itérations.

    history : dict
        Doit contenir une clé "accuracy" avec une liste de valeurs successives.
    """
  
    plt.figure(figsize=(6,4))
    plt.plot(history, marker='o', linewidth=2, color='blue')
    plt.title(title, fontsize=12)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracy_xy(history, title="Accuracy"):
    """
    Affiche l'évolution d'une métrique (ex: accuracy)
    à partir d'une liste de tuples (x, y).
    
    Parameters
    ----------
    history : list of tuples
        Chaque élément est un tuple (x, y)
    title : str
        Titre du graphique
    """
    if not isinstance(history, (list, tuple)) or len(history) == 0:
        print("Erreur : 'history' doit être une liste non vide de tuples (x, y).")
        return

    try:
        x, y = zip(*history)
    except ValueError:
        print("Erreur : les éléments de 'history' doivent être des tuples (x, y).")
        return

    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker='o', linewidth=2, color='blue')
    plt.title(title, fontsize=12)
    plt.xlabel("labels")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

