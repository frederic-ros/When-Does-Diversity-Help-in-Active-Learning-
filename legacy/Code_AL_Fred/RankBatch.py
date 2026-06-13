# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:51:50 2025

@author: frederic.ros
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.base import BaseEstimator
import random
from sklearn.base import BaseEstimator


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from util import plot_accuracy_xy

class PseudoLabeling:
    def __init__(self, n_initial, alpha=0.2, max_iter=20, batch_size=5,save_i=False):
        self.n_initial = n_initial
        self.alpha = alpha  # Pondération par alpha
        self.max_iter = max_iter
        self.batch_size = batch_size  # Nombre d'exemples à sélectionner à chaque itération
        self.D_ESTIMATED = []  # Liste des instances labellisées
        self.y_ESTIMATED = []  # Liste des labels des instances labellisées
        self.U_UNCERTAINTY = []  # Liste des instances non labellisées
        self.score = [(0,0)]
        self.save_i = save_i
    def set_initial_selection(self, X, y_train):
        """
        Sélectionne aléatoirement les premières instances à labelliser.
        """
        indices_initials = np.random.choice(len(X), self.n_initial, replace=False)
        self.D_ESTIMATED = [X[i] for i in indices_initials]
        self.y_ESTIMATED = [y_train[i] for i in indices_initials]
        self.U_UNCERTAINTY = [X[i] for i in range(len(X)) if i not in indices_initials]

    def compute_uncertainty(self, classifier, instance):
        """
        Calcule l'incertitude pour une instance donnée.
        """
        proba = classifier.predict_proba([instance])[0]
        uncertainty = 1 - np.max(proba)  # L'incertitude est l'inverse de la confiance la plus élevée
        return uncertainty

    def select_batch(self, classifier):
        """
        Sélectionne un batch d'instances avec l'incertitude la plus élevée, pondérée par alpha.
        """
        if len(self.U_UNCERTAINTY) == 0:
            return []

        # Calculer l'incertitude pour chaque instance
        uncertainties = [self.compute_uncertainty(classifier, u) for u in self.U_UNCERTAINTY]
        # Mise à jour de U_UNCERTAINTY pour exclure les instances sélectionnées

        # Pondération par alpha (alpha * incertitude)
        weighted_uncertainties = [self.alpha * u for u in uncertainties]

        # Sélectionner les indices du batch avec les incertitudes les plus élevées
        batch_indices = np.argsort(weighted_uncertainties)[-self.batch_size:]
        selected_batch = [self.U_UNCERTAINTY[i] for i in batch_indices]

        return selected_batch

    def fit(self, X, y_train, classifier, X_test, y_test, display = False):
        """
        Entraîne le modèle avec l'algorithme de pseudo-labellisation.
        """
        self.set_initial_selection(X, y_train)
        if display == True: self.display_selected_instances_graph(X, y_train, phase="Initial")

        # Entraîner le classifieur avec les premières instances
        classifier.fit(np.array(self.D_ESTIMATED, dtype=np.float32), np.array(self.y_ESTIMATED, dtype=np.int32))

        # Affichage après la phase initiale
        #self.display_selected_instances_graph(X, y_train, phase="After Initial Selection")

        # Itération du pseudo-labeling
        for iteration in range(self.max_iter):
            if len(self.U_UNCERTAINTY) == 0:
                break

            # Sélectionner un batch d'instances avec l'incertitude la plus élevée
            selected_batch = self.select_batch(classifier)

            if not selected_batch:
                break

            # Ajoute les nouvelles instances et leurs labels à D_ESTIMATED
            for instance in selected_batch:
                # Trouver l'index de l'instance dans les données originales
                instance_idx = np.where(np.all(X == instance, axis=1))[0][0]
                best_label = y_train[instance_idx]

                # Ajouter à D_ESTIMATED
                self.D_ESTIMATED.append(np.array(instance, dtype=np.float32))
                self.y_ESTIMATED.append(int(best_label))

                # Réentraînement du modèle avec les nouveaux labels
                classifier.fit(np.array(self.D_ESTIMATED, dtype=np.float32), np.array(self.y_ESTIMATED, dtype=np.int32))

            # Mise à jour des instances non labellisées
            self.U_UNCERTAINTY = [u for u in self.U_UNCERTAINTY if not any(np.array_equal(u, selected) for selected in selected_batch)]

        
            #print("iter",iteration)
        # Calcul de l'accuracy sur les données de test après l'entraînement
    
        if self.save_i == True: 
            ala, aall = self.calculate_test_accuracy(classifier, X, y_train, X_test, y_test)
        self.score.append((len(self.D_ESTIMATED),ala))
        if display == True: self.display_selected_instances_graph(X, y_train, phase=f"After Iteration {iteration + 1}")
        return ala, aall
    
    def calculate_test_accuracy(self, classifier, X_train, y_train, X_test, y_test):
    
    # --- Première étape : fit avec les données labellisées seulement ---
    # On fit le classifieur avec les données juste labellisées
        classifier.fit(np.array(self.D_ESTIMATED, dtype=np.float32), np.array(self.y_ESTIMATED, dtype=np.int32))

        # Prédictions sur les données de test pour les données juste labellisées
        y_pred_labeled = classifier.predict(X_test)
    
        # Calcul de l'accuracy sur les données de test (avec uniquement les données labellisées)
        accuracy_labeled = accuracy_score(y_test, y_pred_labeled)
        #print(f"Accuracy sur le test avec les données juste labellisées : {accuracy_labeled * 100:.2f}%")

        # --- Deuxième étape : fit avec toutes les données d'entraînement ---
        # On fit le classifieur avec toutes les données d'entraînement (y compris les pseudo-labels)
        classifier.fit(X_train, y_train)

        # Prédictions sur les données de test pour l'ensemble des données d'entraînement
        y_pred_all = classifier.predict(X_test)
    
        # Calcul de l'accuracy sur les données de test avec toutes les données d'entraînement
        accuracy_all = accuracy_score(y_test, y_pred_all)
        #print(f"Accuracy sur le test avec toutes les données d'entraînement : {accuracy_all * 100:.2f}%")
        return accuracy_labeled, accuracy_all
    
    def display_selected_instances_graph(self, X, y_train, phase=""):
        """
        Affiche le graphique des instances sélectionnées à chaque étape.
        """
        plt.figure(figsize=(8, 6))

        # Affichage des données non labellisées en gris
        plt.scatter(X[:, 0], X[:, 1], c='gray', marker='x', alpha=0.5, label='Non Labellisé')

        # Affichage des données initiales (sélectionnées)
        initial_selected = np.array(self.D_ESTIMATED)
        initial_labels = np.array(self.y_ESTIMATED)
    
        # Utilisation de 'o' pour avoir un marqueur rempli et éviter l'avertissement
        plt.scatter(initial_selected[:, 0], initial_selected[:, 1], c=initial_labels, 
                    marker='o', edgecolor='red', label="Données Initiales")
    
        # Affichage des données labellisées
        labeled_selected = np.array(self.D_ESTIMATED)
        labeled_labels = np.array(self.y_ESTIMATED)
        plt.scatter(labeled_selected[:, 0], labeled_selected[:, 1], c=labeled_labels, 
                    marker='o', edgecolor='blue', label="Données Labellisées")

        plt.title(f"Phase: {phase}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(loc="best")
        plt.show()

def display(X, y,title):
    
    plt.scatter(X[:, 0], X[:, 1],  c=y, marker="x")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    
def algoRank(X_train, y_train,X_test, y_test,
             n_initial=2, batch_size=10, alpha=0.7, 
             max_iter=20, classifier=None, display = False, save_i=False):
    
    # Initialisation de la classe de pseudo-labeling
    pseudo_labeler = PseudoLabeling(n_initial=n_initial, batch_size=batch_size, 
                                    alpha=alpha, max_iter=max_iter,save_i = save_i)

    ala, aall = pseudo_labeler.fit(X_train, y_train, classifier, X_test, y_test, display)
    if save_i== True: plot_accuracy_xy(pseudo_labeler.score,"Accuracy RANK")
    return ala

def testi(t):    
    # Exemple de données (1000 instances, 2 features)
    X, y = make_classification(n_samples=2000, class_sep = 1., n_features=4, n_classes=4, 
                                            n_informative=2, n_redundant=0, 
                                            n_clusters_per_class =1,
                                            n_repeated=0, 
                                            random_state=42+t)
    # Split les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    if display == True: display(X_train, y_train, "Init base")
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)

    a1, a2 = algoRank(X_train, y_train,X_test, y_test,n_initial=2, batch_size=10, alpha=0.7, 
             max_iter=20, classifier=classifier)
    print("accuracy",a1, a2)
    '''    
    # Initialisation de la classe de pseudo-labeling
    pseudo_labeler = PseudoLabeling(n_initial=2, batch_size=10, alpha=0.7, max_iter=20)

    # Entraînement avec un classifieur Random Forest
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    pseudo_labeler.fit(X_train, y_train, classifier, X_test, y_test)
    '''
'''    
for i in range(10,11):
    testi(i)
'''    