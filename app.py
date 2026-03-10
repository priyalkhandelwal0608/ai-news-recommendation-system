from flask import Flask, render_template, request
from src.recommender import NewsRecommender
import pandas as pd

app = Flask(__name__)

# Synthetic interactions
interactions = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3],
    'article_id': [1, 2, 2, 3, 4, 5],
    'rating': [5, 3, 4, 5, 2, 4]
})

# Initialize recommender
recommender = NewsRecommender("data/news_dataset.csv", interactions)

@app.route("/", methods=["GET", "POST"])
def index():
    content_results = []
    collab_results = []
    query = ""
    user_id = None

    if request.method == "POST":
        query = request.form.get("query")
        user_id = request.form.get("user_id")

        if query:
            content_results = recommender.content_based(query, top_n=5)
        if user_id:
            try:
                user_id = int(user_id)
                collab_results = recommender.collaborative(user_id, top_n=5)
            except:
                collab_results = []

    return render_template("index.html",
                           content_results=content_results,
                           collab_results=collab_results,
                           query=query,
                           user_id=user_id)

if __name__ == "__main__":
    app.run(debug=True)