import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import json
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))  # Remonte d'un niveau
result_dir = os.path.join(root_dir, "seqstat")
os.makedirs(result_dir, exist_ok=True)
from util import plot_accuracy_xy

def save_sequence(filename, data):
    """Save a list of tuples to a text file in JSON format."""
    with open(filename, 'w') as f:
        json.dump(data, f)

'''
def select_badge_samples(model, X_unlabeled, n_samples):
    """
    Sélectionne des échantillons selon la méthode BADGE.
    
    :param model: Modèle entraîné (ex: un classifieur Scikit-Learn ou un réseau de neurones)
    :param X_unlabeled: Données non labélisées
    :param n_samples: Nombre d'échantillons à sélectionner
    :return: Indices des échantillons sélectionnés
    """
    # Obtenir les prédictions et probabilités
    proba = model.predict_proba(X_unlabeled)  # On suppose que le modèle donne des probabilités

    # Calculer les gradients simulés (proba * features)
    gradients = proba * X_unlabeled  # Approximation naïve

    # Appliquer K-Means++ pour sélectionner des points diversifiés
    kmeans = KMeans(n_clusters=n_samples, init="k-means++", random_state=42)
    kmeans.fit(gradients)

    # Sélectionner les points les plus proches des centroïdes
    selected_indices = np.unique(np.argmin(np.linalg.norm(gradients - kmeans.cluster_centers_[:, None], axis=2), axis=1))

    return selected_indices
'''


class ActiveLearningLoop:
    def __init__(self, X_all, y_all, X_test, y_test,classifier, model_loss, idx_labeled,
                 s, b, linkage='ward', use_badge=False, i_save=0):
        """
        Implémente un apprentissage actif basé sur clustering, avec option d'utiliser BADGE.
        
        :param X_all: Toutes les données.
        :param y_all: Labels complets (labélisés et non labélisés, mais on ne connaît que ceux de idx_labeled).
        :param classifier: Modèle de classification.
        :param model_loss: Fonction qui évalue l'incertitude du modèle.
        :param idx_labeled: Indices initialement labélisés.
        :param s: Taille du sous-ensemble sélectionné pour le clustering.
        :param b: Nombre d'échantillons à labéliser par itération.
        :param linkage: Méthode de liaison pour HAC.
        :param use_badge: Utiliser BADGE pour la sélection des échantillons ?
        """
        self.X_all = X_all
        self.y_all = y_all
        self.X_test = X_test
        self.y_test = y_test
        self.classifier = classifier
        self.model_loss = model_loss
        self.idx_labeled = set(idx_labeled)
        self.idx_unlabeled = set(range(len(X_all))) - self.idx_labeled
        self.s = s
        self.b = b
        self.linkage = linkage
        self.use_badge = use_badge  # Nouvelle option BADGE
        self.i_save = i_save
        self.score =[(0,0)]
    def select_and_label(self):
        """
        Effectue une itération de sélection et mise à jour des labels.
        """
        if len(self.idx_unlabeled) == 0:
            print("Plus d'échantillons non labélisés.")
            return False

        idx_unlabeled_list = list(self.idx_unlabeled)

        # 1️⃣ Entraînement du classifieur
        X_train = self.X_all[list(self.idx_labeled)]
        y_train = self.y_all[list(self.idx_labeled)]
        self.classifier.fit(X_train, y_train)

        # 2️⃣ Prédiction des incertitudes
        X_unlabeled = self.X_all[idx_unlabeled_list]
        lu = self.model_loss(self.classifier, X_unlabeled)

        # 3️⃣ Sélection des s indices les plus incertains
        top_s_indices = np.argsort(-lu)[:self.s]  
        selected_s_idx = [idx_unlabeled_list[i] for i in top_s_indices]

        # 4️⃣ Si BADGE est activé, utiliser cette méthode pour sélectionner des indices
        if self.use_badge:
            # Application de BADGE avec les incertitudes calculées
            selected_idx = self.select_badge_samples(lu, selected_s_idx)
        else:
            # Clustering HAC sur les s échantillons sélectionnés (méthode classique)
            clustering = AgglomerativeClustering(n_clusters=self.b, linkage=self.linkage)
            X_top_s = self.X_all[selected_s_idx]
            cluster_labels = clustering.fit_predict(X_top_s)

            # 5️⃣ Sélection des b points les plus incertains par cluster
            selected_idx = self.select_from_clusters(lu, top_s_indices, selected_s_idx, cluster_labels)

        # 6️⃣ Mise à jour des indices labélisés
        for idx in selected_idx:
            self.idx_labeled.add(idx)
            self.idx_unlabeled.remove(idx)

        return True  # L'itération a bien eu lieu

    def select_from_clusters(self, lu, top_s_indices, selected_s_idx, cluster_labels):
        """
        Sélectionne les b points les plus incertains dans chaque cluster.
        """
        selected_indices = []
        for i in range(self.b):
            cluster_mask = (cluster_labels == i)
            cluster_lu = lu[top_s_indices][cluster_mask]
            if len(cluster_lu) > 0:
                max_loss_idx = np.argmax(cluster_lu)  # Trouver l'indice local dans le cluster
                true_idx = np.where(cluster_mask)[0][max_loss_idx]  # Trouver l'indice global dans top_s_indices
                selected_indices.append(selected_s_idx[true_idx])  # Sélectionner le bon élément
        return selected_indices


    def select_badge_samples(self, lu, selected_s_idx):
        """
        Utilise la méthode BADGE pour sélectionner des échantillons avec la plus grande incertitude et diversité.
        """
        gradients = np.abs(lu)  # Utilisation de l'incertitude comme "marge"

        # Trouver les indices relatifs dans l'ensemble des non-labélisés
        unlabeled_indices = list(self.idx_unlabeled)  # Les indices non labélisés dans X_all

        # Appliquer un clustering K-means++ sur les gradients des non-labélisés
        kmeans = AgglomerativeClustering(n_clusters=self.b, linkage=self.linkage)

        # Appliquer l'algorithme de clustering sur les gradients des non-labélisés
        cluster_labels = kmeans.fit_predict(gradients.reshape(-1, 1))  # Appliquer sur les incertitudes des non-labélisés

        selected_idx = []
        for i in range(self.b):
            # Pour chaque cluster, on sélectionne le point avec la plus grande incertitude
            cluster_mask = (cluster_labels == i)

            # Sélectionner les gradients des non-labélisés dans ce cluster
            cluster_lu = gradients[cluster_mask]

            if len(cluster_lu) > 0:
                max_loss_idx = np.argmax(cluster_lu)

                # Traduction de l'indice du non-labélisé dans le référentiel des non-labélisés
                selected_unlabeled_idx = np.where(cluster_mask)[0][max_loss_idx]

                # Traduction de l'indice sélectionné dans le référentiel global
                # On utilise le tableau des indices non-labélisés pour retrouver l'indice global
                selected_idx.append(unlabeled_indices[selected_unlabeled_idx])

        return selected_idx
    
    def run(self, max_iter=10):
        """
        Exécute plusieurs itérations d'apprentissage actif.
        """
        acc=0
        for i in range(max_iter):
            #print(f"⚡ Itération {i+1}")
            if not self.select_and_label():
                print("✅ Fin de l'apprentissage actif.")
                break
            else:
                labels = list(self.idx_labeled)
                self.classifier.fit(self.X_all[labels], self.y_all[labels])
                if self.i_save==True: 
                    acc = self.classifier.score(self.X_test,self.y_test)
                self.score.append((len(labels),acc))
        '''        
        filei = "data" + str(self.i_save) + "/"      
        file_path = os.path.join(result_dir,filei+ "rank2022" + str(self.i_save)+ ".txt")
        save_sequence(file_path, x)            
        '''
        
def visualize_labeled_points(X,X_initial, X_final, y_initial, y_final):
    """
    Visualise les points initialement labellisés et ceux labellisés après l'algorithme.

    - X_initial, y_initial : Points labellisés au début (en rouge)
    - X_final, y_final : Tous les points labellisés après exécution (en bleu)
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(X[:, 0], X[:, 1], color="grey", alpha=0.6, label="initial data")

    # Affichage des points finaux labellisés
    plt.scatter(X_final[:, 0], X_final[:, 1], c=y_final, cmap='coolwarm', alpha=0.6, label="Final Labeled")

    # Affichage des points initialement labellisés
    plt.scatter(X_initial[:, 0], X_initial[:, 1], c=y_initial, cmap='coolwarm', edgecolors='k', label="Initial Labeled")

    plt.legend()
    plt.title("RANK2022")
    plt.show()


def visualize_data(X,y):
    """
    Visualise les points initialement labellisés et ceux labellisés après l'algorithme.

    - X_initial, y_initial : Points labellisés au début (en rouge)
    - X_final, y_final : Tous les points labellisés après exécution (en bleu)
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, label="Final Labeled")

    
    plt.legend()
    plt.title("data set")
    plt.show()
# Fonction de prédiction de l'incertitude (score basé sur la confiance)
def model_loss(classifier, X):
    probs = classifier.predict_proba(X)
    return -np.max(probs, axis=1)  # Plus incertain = probabilité max la plus faible
    
def rank2022(X_train, X_test, y_train, y_test,init,use_badge = False,
             s=50, b=10, classifier = SVC(probability=True, random_state=42),
             iteration=10, i_save=0,
             display = False):
    # Séparation des données en un ensemble initialement labélisé et non labélisé
    idx_labeled_init = np.random.choice(len(X_train), init, replace=False)  # 10 indices initialement labélisés

    # Création de l'instance d'apprentissage actif
    active_learner = ActiveLearningLoop(X_train, y_train, X_test, y_test, classifier, 
                                    model_loss, idx_labeled_init, 
                                    s=s, b=b,
                                    linkage='ward', use_badge=use_badge, i_save=i_save)

    # Exécution des itérations
    active_learner.run(max_iter=iteration)


    X_labeled_initial = X_train[idx_labeled_init]
    y_labeled_initial = y_train[idx_labeled_init]
    X_labeled_final = X_train[list(active_learner.idx_labeled)]  
    y_labeled_final = y_train[list(active_learner.idx_labeled)]  
    
    if display==True: visualize_labeled_points(X_train,X_labeled_initial, 
                         X_labeled_final, y_labeled_initial, 
                         y_labeled_final)



    # 1️⃣ Entraînement sur les données labélisées issues de l'Active Learning
    clf_active = active_learner.classifier  # On utilise le même modèle que dans active learning
    clf_active.fit(X_train[list(active_learner.idx_labeled)], y_train[list(active_learner.idx_labeled)])
    y_pred_active = clf_active.predict(X_test)
    acc_active = accuracy_score(y_test, y_pred_active)
    '''
    # 2️⃣ Entraînement sur l'ensemble des données `X_train, y_train`
    clf_full = active_learner.classifier.__class__()  # Instanciation d'un nouveau modèle identique
    clf_full.fit(X_train, y_train)
    y_pred_full = clf_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    '''
    if i_save==True: plot_accuracy_xy(active_learner.score, title="Accuracy RANK2022")
    # Affichage des résultats
    #print(f"{len(active_learner.idx_labeled)}🔹 Acc (Full, Active) : {acc_full:.4f},{acc_active:.4f}")
    #    return acc_full, acc_active
    return acc_active

def test(i):    
   

    n_samples = 10000
    n_features = 2
    random_state = 42
    n_classes = 4
    n_clusters_per_class = 1
    class_sep = 0.8
    use_badge = True
    iteration = 20
    s=50
    b=10
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_features, n_redundant=0, 
                               n_repeated=0, n_classes=n_classes,
                               n_clusters_per_class =n_clusters_per_class, 
                               class_sep = class_sep, random_state=random_state+i)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    visualize_data(X_train, y_train)

    init = 10
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    #classifier = SVC(probability=True, random_state=42)
    acc_full, acc_active = rank2022(X_train, X_test, y_train, y_test,init,
                                    s=s, b=b, use_badge = use_badge,
                                    classifier = classifier,iteration=iteration)
    print(f"🔹 Acc (Full, Active) : {acc_full:.4f},{acc_active:.4f}")
'''
for i in range(0,5):
    test(i)
'''    