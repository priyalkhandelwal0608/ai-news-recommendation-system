from flask import Flask, render_template, request
from src.recommender import NewsRecommender
import pandas as pd

app = Flask(__name__)

# Synthetic interactions
interactions = pd.DataFrame({
    'user_id': [
        1,1,2,2,3,3,4,4,5,5,
        6,6,7,7,8,8,9,9,10,10,
        11,11,12,12,13,13,14,14,15,15,
        16,16,17,17,18,18,19,19,20,20,
        21,21,22,22,23,23,24,24,25,25
    ],
    'article_id': [
        1,2,2,3,4,5,1,3,2,6,
        7,8,3,9,4,10,5,11,6,12,
        7,13,8,14,9,15,10,16,11,17,
        12,18,13,19,14,20,15,21,16,22,
        17,23,18,24,19,25,20,1,21,2
    ],
    'rating': [
        5,3,4,5,2,4,3,5,4,2,
        5,3,4,2,5,4,3,5,2,4,
        5,3,4,5,2,4,3,5,4,2,
        5,3,4,2,5,4,3,5,2,4,
        5,3,4,5,2,4,3,5,4,2
    ]
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
