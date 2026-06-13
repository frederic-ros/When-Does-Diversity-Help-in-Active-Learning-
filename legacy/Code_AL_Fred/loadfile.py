# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 12:46:40 2025

@author: frederic.ros
"""
import sys
import os
import pandas as pd

def load_classification_data_from_directory(directory_path):
    data_list = []
    labels_list = []
    files = []
    sizes = []
    p = []

    # Parcourir tous les fichiers dans le répertoire

    for filename in os.listdir(directory_path):

        if filename.endswith(".txt"): # Vérifier que le fichier est un .txt
            file_path = os.path.join(directory_path, filename)
            print("file found:",filename)
            # Lire le fichier avec pandas
            df = pd.read_csv(file_path, delimiter="\t", header=None) # Ajuster le délimiteur si nécessaire

            # Séparer les données (colonnes sauf la dernière) et les labels (dernière colonne)
            data = df.iloc[:, :-1].values # Convertir les données en array
            labels = df.iloc[:, -1].values # Convertir les labels en array
            # Ajouter les données et les labels aux listes

            data_list.append(data)
            labels_list.append(labels)
            files.append(filename)
            sizes.append(len(data))
            p.append(data.shape[1])

    return files, sizes, p, data_list, labels_list
