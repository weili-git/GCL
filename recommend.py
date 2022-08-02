import numpy as np
import matplotlib as plt
import pandas as pd

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

data = pd.merge(ratings, movies, on='movieId')  # join

ratings_average = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings_average['ratings_count'] = pd.DataFrame(data.groupby('title')['rating'].count())

ratings_matrix = data.pivot_table(index='userId', columns='title', values='rating')

favorite_movie_ratings = ratings_matrix['Waiting to Exhale (1995)']

similar_movies = ratings_matrix.corrwith(favorite_movie_ratings)

correlation = pd.DataFrame(similar_movies, columns=['Correlation'])
correlation.dropna(inplace=True)

correlation = correlation.join(ratings_average['ratings_count'])

recommendation = correlation[correlation['ratings_count']>100].sort_values('Correlation', ascending=False)
recommendation = recommendation.merge(movies, on='title')

print(recommendation.head(10))
