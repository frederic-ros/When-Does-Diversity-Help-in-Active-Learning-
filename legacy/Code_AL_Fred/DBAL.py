import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import json
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL*")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))  # Remonte d'un niveau
result_dir = os.path.join(root_dir, "seqstat")
os.makedirs(result_dir, exist_ok=True)
from util import plot_accuracy, plot_accuracy_xy

classifiers = {
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}
def save_sequence(filename, data):
    """Save a list of tuples to a text file in JSON format."""
    with open(filename, 'w') as f:
        json.dump(data, f)

class DBAL:
    def __init__(self, X_train, y_train, X_test, y_test, base_model, budget,
                 batch_size, prefilter_factor,i_save=0):
        self.base_model = base_model
        self.budget = budget  # Nombre total d'items à ajouter
        self.batch_size = batch_size  # Nombre d'items ajoutés à chaque itération
        self.prefilter_factor = prefilter_factor  # Facteur de sélection avant clustering
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.i_save = i_save
        self.score = [(0,0)]

    def fit(self, X_labeled, y_labeled, X_unlabeled, oracle):
        """ Exécute l'algorithme DBAL avec un oracle externe pour obtenir les labels. """
        self.model = clone(self.base_model)
        self.model.fit(X_labeled, y_labeled)

        # Données non labellisées (Deep Unlabeled Pool)
        DUL = X_unlabeled.copy()

        # Stocke les exemples labellisés au fur et à mesure
        Xt, yt = [X_labeled], [y_labeled]

        budget_used = 0
        
        
        acc = self.model.score(self.X_test, self.y_test)
        self.score.append((len(X_labeled),acc))
        while budget_used < self.budget and len(DUL) > 0:
            # 1. Estimation des incertitudes
            probas = self.model.predict_proba(DUL)
            uncertainties = 1 - np.max(probas, axis=1)

            # 2. Pré-sélection des exemples les plus informatifs
            top_k = min(self.prefilter_factor * self.batch_size, len(DUL))
            selected_indices = np.argsort(uncertainties)[-top_k:]

            X_prefiltered = DUL[selected_indices]

            # 3. Clustering pour diversité (K-Means)
            kmeans = KMeans(n_clusters=self.batch_size, n_init=10, random_state=42)
            kmeans.fit(X_prefiltered)
            cluster_centers = kmeans.cluster_centers_

            # 4. Sélection des points les plus proches des centroïdes
            selected_final = []
            for center in cluster_centers:
                distances = np.linalg.norm(X_prefiltered - center, axis=1)
                selected_final.append(selected_indices[np.argmin(distances)])

            X_new = DUL[selected_final]

            # **5. Appel à l'oracle pour obtenir les vrais labels**
            y_new = oracle(X_new,self.X_train, self.y_train)

            # 6. Mise à jour des données labellisées
            Xt.append(X_new)
            yt.append(y_new)

            # Mise à jour de DUL (suppression des éléments sélectionnés)
            DUL = np.delete(DUL, selected_final, axis=0)

            # 7. Ré-entraînement du modèle
            X_train = np.vstack(Xt)
            y_train = np.hstack(yt)

            self.model = clone(self.base_model)
            self.model.fit(X_train, y_train)

            budget_used += self.batch_size
        
            if self.i_save==True: 
                acc = self.model.score(self.X_test, self.y_test)
            self.score.append((len(X_train),acc))
        '''
        filei = "data" + str(self.i_save) + "/"    
        file_path = os.path.join(result_dir, filei+ "dbal"+str(self.i_save)+".txt")
        save_sequence(file_path, score)    
        '''
            

        return np.vstack(Xt), np.hstack(yt)


def plot_data(X_t, y_t ):
        """Affiche l'évolution de la pseudo-labélisation (limité à 2D)"""
        plt.figure(figsize=(6, 6))
    
        # Affichage des points non labellisés
        
        plt.scatter(X_t[:, 0], X_t[:, 1], c=y_t, alpha=0.3, label="data")
        
        
        plt.legend()
        plt.show()

def oracle(X_selected, X_train, y_train):
    """ Oracle simulé : retourne les vrais labels """
    indices = [np.where((X_train == x).all(axis=1))[0][0] for x in X_selected]
    return y_train[indices]

def Visualize(X_train, X_final, y_final, X_labeled):
# **Visualisation**
    plt.scatter(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.3, label="Unlabeled")
    plt.scatter(X_final[:, 0], X_final[:, 1], c=y_final, cmap='viridis', edgecolors='k', label="Labeled")
    plt.scatter(X_labeled[:, 0], X_labeled[:, 1], marker='x', color='red', label="Initial Labeled")
    plt.title("DBAL")
    plt.legend()
    plt.show()
    
def Visualize_prediction(X_test, y_test, y_predict, X_labeled=None):
    """ Visualisation des labels réels et prédits côte à côte. """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Première figure : y_test (labels réels)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k', label="True Labels")
    if X_labeled is not None:
        axes[0].scatter(X_labeled[:, 0], X_labeled[:, 1], marker='x', color='red', label="Initial Labeled")
    axes[0].set_title("True Labels")
    axes[0].legend()

    # Deuxième figure : y_predict (labels prédits)
    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_predict, cmap='viridis', edgecolors='k', label="Predicted Labels")
    if X_labeled is not None:
        axes[1].scatter(X_labeled[:, 0], X_labeled[:, 1], marker='x', color='red', label="Initial Labeled")
    axes[1].set_title("Predicted Labels")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def DBAL_algo(X_train, X_test, y_train, y_test,budget=100,n_init = 10,
              batch=10,classifier = RandomForestClassifier(n_estimators=50,
                                                           random_state=42), 
              i_save=0, display=False):

    #plot_data(X_train, y_train)
    if classifier is None:
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
    elif isinstance(classifier, (SVC, RandomForestClassifier, LogisticRegression, KNeighborsClassifier)):
        base_model = classifier
    
    # Séparation des données labellisées et non labellisées
    initial_labeled = n_init  # Nombre d'éléments initialement labellisés
    initial_labeled_indices = np.random.choice(len(X_train), size=n_init, replace=False)
    X_labeled, y_labeled = X_train[initial_labeled_indices], y_train[initial_labeled_indices]
    all_indices = set(range(len(X_train)))
    # Indices non labélisés
    unlabeled = all_indices - set(initial_labeled_indices)
    X_unlabeled = X_train[np.array(list(unlabeled))]
    
    '''
    X_labeled, y_labeled = X_train[:initial_labeled], y_train[:initial_labeled]
    print(y_train[:initial_labeled])
    X_unlabeled = X_train[initial_labeled:]
    '''
    
    #budget = (int)(2 * len(X_train) / 100)

    # 🔹 **Exécution de l'algorithme**

    dbal = DBAL(X_train, y_train, X_test, y_test,base_model, budget=budget, batch_size=batch,
                prefilter_factor=3, i_save=i_save)
    X_final, y_final = dbal.fit(X_labeled, y_labeled, X_unlabeled, oracle)

    # Évaluation de l'accuracy
    model_test = clone(base_model)
    model_test.fit(X_final, y_final) #on fit avec les X_final et y_final
    y_pred = model_test.predict(X_test) #on predit les X_tests
    accuracy = accuracy_score(y_test, y_pred)

#    print(f"Accuracy après DBAL: {accuracy:.4f}", "ratio", 100*len(y_final)/len(y_train))
    '''
    # Entraînement sur l'ensemble du jeu de données pour comparaison
    model_test.fit(X_train, y_train)
    y_pred_full = model_test.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    print(f"Accuracy en utilisant toute la base de training: {acc_full:.4f}")
    '''
    if display==True: Visualize(X_train, X_final, y_final, X_labeled)
    if i_save== True: plot_accuracy_xy(dbal.score,"Accuracy DBAL")
    #Visualize_prediction(X_test, y_test, y_pred)
    return accuracy

# Génération d'un jeu de données synthétique
# Génération d'un jeu de données synthétique
'''
for i in range(1):
    X, y = make_classification(n_samples=10000, class_sep = 1.,n_classes=2, n_features=2, n_informative=2, 
                               n_redundant=0, n_clusters_per_class=2, random_state=41+i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    DBAL_algo(X_train, X_test, y_train, y_test)
'''