import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class NewsRecommender:
    def __init__(self, articles_path, interactions=None):
        self.articles = pd.read_csv(articles_path)
        self.interactions = interactions
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.articles['content'])

    def content_based(self, query, top_n=5):
        user_vec = self.tfidf.transform([query])
        cosine_sim = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[-top_n:][::-1]
        return self.articles.iloc[top_indices][['title', 'category']].to_dict(orient='records')

    def collaborative(self, user_id, top_n=5):
        if self.interactions is None:
            return []
        user_item_matrix = self.interactions.pivot_table(index='user_id', columns='article_id', values='rating').fillna(0)
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(user_item_matrix)
        user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = model.kneighbors(user_vector, n_neighbors=3)
        similar_users = user_item_matrix.iloc[indices[0]].index
        recommended_articles = self.interactions[self.interactions['user_id'].isin(similar_users)]['article_id'].unique()
        return self.articles[self.articles['article_id'].isin(recommended_articles)][['title', 'category']].to_dict(orient='records')