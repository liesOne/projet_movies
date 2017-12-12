#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:10:03 2017

@author: lies
"""

from flask import Flask
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

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
    
data_dummies = pd.read_csv("data_dummies.csv", sep = ";")
data_movies = pd.read_csv("data_movies.csv", sep = ";")
    
app = Flask(__name__)

@app.route('/recommend/<titre_film>')
def reco_film(titre_film):
    liste_film = preco_film(titre_film, data_movies, data_dummies)
    return mise_en_forme(liste_film)
