# AI News Recommendation System (without Surprise)
# Libraries: sklearn, pandas, numpy

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# 1. Sample Dataset
# -----------------------------
articles = pd.DataFrame({
    'article_id': [1, 2, 3, 4, 5],
    'title': [
        "AI beats humans in chess",
        "Stock markets hit record high",
        "New breakthrough in cancer research",
        "Football World Cup highlights",
        "Climate change impacts agriculture"
    ],
    'content': [
        "Artificial intelligence systems are now beating grandmasters in chess.",
        "Global stock markets reached an all-time high due to tech growth.",
        "Scientists discovered a new protein that may help fight cancer.",
        "The World Cup saw thrilling matches and unexpected results.",
        "Climate change is reducing crop yields and affecting farmers worldwide."
    ],
    'category': ["Technology", "Finance", "Health", "Sports", "Environment"]
})

interactions = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3],
    'article_id': [1, 2, 2, 3, 4, 5],
    'rating': [5, 3, 4, 5, 2, 4]
})

# -----------------------------
# 2. Content-Based Filtering
# -----------------------------
def content_based_recommend(user_profile_text, articles, top_n=3):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(articles['content'])
    
    user_vec = tfidf.transform([user_profile_text])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    return articles.iloc[top_indices][['title', 'category']]

print("Content-Based Recommendations:")
print(content_based_recommend("AI technology innovation", articles))

# -----------------------------
# 3. Collaborative Filtering (Sklearn Nearest Neighbors)
# -----------------------------
def collaborative_filtering(interactions, articles, user_id, top_n=3):
    # Create user-item matrix
    user_item_matrix = interactions.pivot_table(index='user_id', columns='article_id', values='rating').fillna(0)
    
    # Fit Nearest Neighbors
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)
    
    # Find nearest neighbors for the user
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=3)
    
    # Get articles liked by similar users
    similar_users = user_item_matrix.iloc[indices[0]].index
    recommended_articles = interactions[interactions['user_id'].isin(similar_users)]['article_id'].unique()
    
    return articles[articles['article_id'].isin(recommended_articles)][['title', 'category']]

print("\nCollaborative Filtering Recommendations for User 1:")
print(collaborative_filtering(interactions, articles, user_id=1))

# -----------------------------
# 4. Hybrid Recommendation
# -----------------------------
def hybrid_recommend(user_id, user_profile_text, articles, interactions, top_n=3):
    # Content-based scores
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(articles['content'])
    user_vec = tfidf.transform([user_profile_text])
    content_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # Collaborative scores (average rating from similar users)
    user_item_matrix = interactions.pivot_table(index='user_id', columns='article_id', values='rating').fillna(0)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=3)
    similar_users = user_item_matrix.iloc[indices[0]].index
    collab_scores = articles['article_id'].apply(
        lambda aid: interactions[(interactions['user_id'].isin(similar_users)) & (interactions['article_id']==aid)]['rating'].mean()
    ).fillna(0).values
    
    # Hybrid: weighted sum
    hybrid_scores = 0.5 * content_scores + 0.5 * collab_scores
    top_indices = hybrid_scores.argsort()[-top_n:][::-1]
    return articles.iloc[top_indices][['title', 'category']]

print("\nHybrid Recommendations for User 1:")
print(hybrid_recommend(1, "AI technology innovation", articles, interactions))