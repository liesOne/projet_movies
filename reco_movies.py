# coding: utf-8

# **Importation des bibliothèques**
import pandas as pd
import numpy as np

# **Chargement de la base de données**
#Chargement des données
data_origine = pd.read_csv('movie_metadata.csv')
#On conserve la base de données d'origine
data = data_origine.copy()

# <h2>**A - ANALYSE EXPLORATOIRE DES DONNEES**</h2>
#Création d'une fonction pour determiner la proportion de cellules non renseignés('NaN') par colonne
def calcul_taux_nan_colonne(df):
    '''
    df: est un DataFrame
    Calcul le taux de NaN par colonne d'un DataFrame.
    La fonction retourne un DataFrame avec en lignes les libellés colonnes
    et une colonne correspond au taux de NaN dans l'ordre décroissant.
    '''
    dict_nan = {}
    nbre_lignes = len(df)
    for col in df.columns:
        nbre_nan = df[col].isnull().sum()
        dict_nan[col] = round(nbre_nan / nbre_lignes * 100, 1)
    df1 = pd.DataFrame(dict_nan, index = ['Taux NaN']).transpose().sort_values('Taux NaN', ascending = False)
    return df1

# **<h2>Analyse croisée des colonnes quantitatives</h2>**

# **Création d'une liste avec uniquement les données quantitatives et remplacement des 'NaN' par la valeur médiane de chaque colonne**
#Création d'une variable pour conserver les colonnes quantitatives
#et calcul de la mediane pour les valeurs manquantes
data_quanti = data.select_dtypes(include=['float64', 'int64'])

#Chargement d'une bibliothèque de SciKit Learn permettant de completer les valeurs manquantes
from sklearn.preprocessing import Imputer
#Choix de remplir les valeurs manquantes par la valeur médiane (paramètrages)
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
#Application des paramétrages sur les données
data_quanti = pd.DataFrame(imp.fit_transform(data_quanti), columns = data_quanti.columns, index = data_quanti.index)
data[data_quanti.columns] = data_quanti

# **Calcul du nombre de NaN par ligne**

#Création d'une fonction pour calculer le taux de 'NaN' par ligne d'une base de donnée
def calcul_taux_nan_ligne(df):
    '''
    df: un DataFrame
    renvoie une colonne de type DataFrame contenant le taux de NaN
    de chaque ligne du DataFrame fourni
    '''
    dict_nan = {}
    for index in df.index:
        nbre_col = len(df.columns)
        nbre_nan = df.loc[index].isnull().sum()
        dict_nan[index] = round(nbre_nan / nbre_col * 100, 1)
    df1 = pd.DataFrame(dict_nan, index = ['Taux NaN ligne']).T
    return df1

df_taux_nan_ligne = calcul_taux_nan_ligne(data)
df_taux_nan_ligne['Taux nan ligne 2'] = df_taux_nan_ligne
nbre_ligne_nan = df_taux_nan_ligne.groupby('Taux NaN ligne').count()

# **Identification des films en doublons**

nbre_lignes_avant = len(data)
data = data.drop_duplicates(['director_name', 'movie_title', 'actor_1_name'])
'Nombre de doublons supprimé: {}'.format(nbre_lignes_avant - len(data))

# **Calcul des occurences de la colonne 'genres'**

#Création d'une fonction permettant de lister les mots et de les dénombrer
def dico_des_occurences(df, sep = '|'):
    '''
    df: un DataFrame
    sep: le séparateur
    Renvoi un dictionnaire avec en clé le mot et en valeur le nombre d'occurrence
    '''
    dict_occurence = {}
    for index, row in df.iteritems():
        try:
            for mot in row.split(sep):
                if mot in dict_occurence:
                    dict_occurence[mot] += 1
                else:
                    dict_occurence[mot] = 1
        except:
            continue
    return dict_occurence

dict_occu = dico_des_occurences(data["genres"])

#Transformation du dico en liste
list_occu = [[cle, valeur] for cle, valeur in dict_occu.items()]
#Tri inverse de la liste en fct de la 2eme colonne
list_occu = sorted(list_occu, key = lambda col: col[1], reverse = True)

# Création de 26 colonnes correspondant à chaque genre et suppression de la colonne "genre"
#Suppression des colonnes Films-Noir, Short, News, Reality-TV et Game-Show
cle_suppr = []
for cle, val in dict_occu.items():
    if val <10:
        cle_suppr.append(cle)
for cle in cle_suppr:
    del(dict_occu[cle])
mot_col_genre = dict_occu.keys()
for genre in mot_col_genre:
    data['genre' + '_' + genre] = pd.DataFrame(np.zeros(len(data)), index = data.index)
for col in data.columns:
    if 'genre_' in col:
        data[col] = data['genres'].apply(lambda x: 1 if col[6:] in x else 0)
del(data['genres'])

# **Calcul des occurences de la colonne 'plot_keywords'**

mot_col_plot_keywords = dico_des_occurences(data['plot_keywords'])
df_occurence_plot_keywords = pd.DataFrame(mot_col_plot_keywords, index =['Nbre occurence']).T.sort_values('Nbre occurence', ascending = False)
df_occurence_plot_keywords['pourcentage'] = round(df_occurence_plot_keywords['Nbre occurence'] / df_occurence_plot_keywords['Nbre occurence'].sum() * 100, 1)


# **Retraitement des noms d'acteur**

#Je créé un nouveau DataFrame avec les 3 colonnes acteurs
data_acteur = data[['actor_1_name','actor_2_name','actor_3_name']]


#Création d'un dictionnaire avec en clé l'acteur, et en valeur le nombre d'apparition dans les 3 colonnes
dico_acteur_2 = {}
for col in data_acteur.columns:
    dico_acteur = dico_des_occurences(data_acteur[col])
    for acteur, nbr in dico_acteur.items():
        if acteur in dico_acteur_2:
            dico_acteur_2[acteur] += dico_acteur[acteur]
        else:
            dico_acteur_2[acteur] = dico_acteur[acteur]

#création d'une fonction renvoyant l'acteur dans le plus fréquent sur les trois acteur fournit en argument
def acteur(*x):
    max_nbre = 0
    max_acteur = ''
    for i in x:
        if type(i).__name__ == 'str':
            if dico_acteur_2[i] > max_nbre:
                max_nbre = dico_acteur_2[i]
                max_acteur = i
    return str(max_acteur)

#Création de la colonne 'actor_name'
data_acteur['actor_name'] = data_acteur.apply(lambda row: acteur(*row), axis = 1)

#On intégre cette nouvelle colonne dans le base de donnée d'origine
data['actor_name'] = data_acteur['actor_name']

#On visualise le résultat de cette nouvelle colonne par curiosité
dico_acteur = dico_des_occurences(data['actor_name']); dico_acteur
liste_acteur = list(zip(dico_acteur.keys(), dico_acteur.values()))
liste_acteur = sorted(liste_acteur, key = lambda x: x[1], reverse = True); liste_acteur


# Liste des directeurs de film:

dico_realisateur = dico_des_occurences(data['director_name'])

# **suppression des colonnes non utile à l'analyse**

col_supprime = ['color', 'num_critic_for_reviews', 'duration', 'director_facebook_likes',                 'actor_3_facebook_likes', 'actor_2_name', 'actor_1_facebook_likes', 'gross', 'actor_1_name',                 'movie_title', 'num_voted_users', 'cast_total_facebook_likes', 'actor_3_name',                 'facenumber_in_poster', 'plot_keywords', 'movie_imdb_link', 'num_user_for_reviews',                 'language', 'country', 'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes',                 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']

data_1 = data.drop(col_supprime, axis = 1)

# <h2>**B - RECHERCHE D'UN MODELE DE CLASSIFICATION**</h2>

# **DUMMIES VARIABLE - Transformation des données qualitatives en données quantitatives**

data_dummies = pd.get_dummies(data_1)

#suppression des colonnes pour lesquelles l'acteur ou le réalisateur n'apparaissent qu'une fois
for col in data_dummies.columns:
    if 'actor_name_' in col:
        if col[11:] != '' and dico_acteur[col[11:]] < 2:
            del(data_dummies[col])
    else:
        if 'director_name_' in col:
            if col[14:] != '' and dico_realisateur[col[14:]] < 2:
                del(data_dummies[col])

# **CENTRAGE DES DONNEES**

from sklearn.preprocessing import StandardScaler
#Centrage des données mais pas de normalisation
scaler = StandardScaler(with_mean = True, with_std = False)
data_dummies_scaled = scaler.fit_transform(data_dummies)
#Je transforme data_dummies_scaled en DataFrame (auparavant un Array Numpy)
data_dummies_scaled = pd.DataFrame(data_dummies_scaled, index = data_dummies.index, columns = data_dummies.columns)

# **ACP  - Analyse en Composantes Principales**

from sklearn.decomposition import PCA
pca = PCA()
pca = PCA(n_components = 20)
data_pca = pd.DataFrame(pca.fit_transform(data_dummies_scaled), index = data_dummies_scaled.index)


# <h2>**CAH - Classification ascendante hierarchique**</h2>

#librairies pour la CAH
#générer la matrice des liens
from scipy.cluster.hierarchy import linkage, fcluster
Z = linkage(data_pca, method = 'ward', metric ='euclidean')
max_d = 3.9
labels_hca = pd.DataFrame(fcluster(Z, max_d, criterion = 'distance'), columns = ['HCA'], index = data_dummies_scaled.index)


labels_hca = pd.DataFrame(fcluster(Z, max_d, criterion = 'distance'), columns = ['HCA'], index = data_dummies_scaled.index)

# <h2>**K-MEANS VS CAH**</h2>
data_movies = data[['movie_title']]
data_movies['HCA'] = labels_hca['HCA']

from sklearn.metrics.pairwise import euclidean_distances
#création d'une fonction pour calculer une distance entre 2 films
def distance(f1, f2):
    distance = euclidean_distances(f1, f2)
    return distance

def preco_film(titre_film, data_movies, data_dummies):
    data_movies['flag'] = data_movies['movie_title'].apply(lambda x: titre_film in x)
    index_film = (data_dummies[data_movies['flag'] == 1].index)[0]
    index_cluster = data_movies['HCA'].loc[index_film]
    data_dummies = data_dummies[data_movies['HCA'] == index_cluster]
    film_1 = data_dummies.loc[index_film].values.reshape(1, -1)
    liste_film = []
    for i, row in data_dummies.iterrows():
        if i == index_film:
            liste_film.append([data_movies['movie_title'][i], -1])
        else:
            film_2 = (data_dummies.loc[i]).values.reshape(1, -1)
            liste_film.append([data_movies['movie_title'][i], float(distance(film_1, film_2))])
    film_trie = sorted(liste_film, key = lambda x: x[1])
    return [i for i, j in film_trie[1:6]]

def mise_en_forme(liste):
    x = "<br>".join(liste)
    return x 

from flask import Flask
app = Flask(__name__)

@app.route('/recommend/<titre_film>')
def reco_film(titre_film):
    liste_film = preco_film(titre_film, data_movies, data_dummies)
    return mise_en_forme(liste_film)

if __name__=='__main__':
    app.run(debug = True)

