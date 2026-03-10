import pandas as pd

def load_interactions():
    # Synthetic user interactions (you can expand this)
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3],
        'article_id': [1, 2, 2, 3, 4, 5],
        'rating': [5, 3, 4, 5, 2, 4]
    })