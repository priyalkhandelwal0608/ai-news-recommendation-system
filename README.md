# AI News Recommendation System 📰🤖

A high-performance, hybrid news recommendation engine that leverages Natural Language Processing (NLP) and Collaborative Filtering to deliver personalized content. Built with **Flask**, **Scikit-Learn**, and **Pandas**.

---

## 🌟 Features

- **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering to provide diverse and accurate suggestions.
- **Content-Based Filtering**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to match articles based on text semantics and search queries.
- **Collaborative Filtering**: Implements **K-Nearest Neighbors (KNN)** to find similar user behavior patterns and suggest "trending" content within specific user clusters.
- **Dynamic Web Interface**: A responsive dashboard to search for articles and view personalized "For You" recommendations.
- **Synthetic Data Integration**: Pre-loaded with interaction data to demonstrate collaborative filtering logic out of the box.

---

## 📂 Project Structure

```text
AI-NEWS-RECOMMENDATION-SYSTEM/
├── data/
│   └── news_dataset.csv       # Dataset containing article titles, categories, and content
├── src/
│   ├── recommender.py         # Core engine: TF-IDF vectorization and KNN logic
├── static/
│   └── style.css              # Custom UI styling and layout
├── templates/
│   └── index.html             # Flask Jinja2 template for the web dashboard
├── app.py                     # Flask server and routing logic
└── requirements.txt           # Project dependencies

---
##🛠️ Tech Stack
Backend: Python 3.x, Flask

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (TfidfVectorizer, NearestNeighbors, Cosine Similarity)

Frontend: HTML5, CSS3, Google Fonts
