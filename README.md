# AI News Recommendation System 

A high-performance, hybrid news recommendation engine that leverages Natural Language Processing (NLP) and Collaborative Filtering to deliver personalized content. Built with **Flask**, **Scikit-Learn**, and **Pandas**.

---

##  Features

- **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering to provide diverse and accurate suggestions.
- **Content-Based Filtering**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to match articles based on text semantics and search queries.
- **Collaborative Filtering**: Implements **K-Nearest Neighbors (KNN)** to find similar user behavior patterns and suggest "trending" content within specific user clusters.
- **Dynamic Web Interface**: A responsive dashboard to search for articles and view personalized "For You" recommendations.
- **Synthetic Data Integration**: Pre-loaded with interaction data to demonstrate collaborative filtering logic out of the box.

---
---

## 🛠️ Tech Stack

* **Backend:** Python 3.x, Flask
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (TfidfVectorizer, NearestNeighbors, Cosine Similarity)
* **Frontend:** HTML5, CSS3, Google Fonts
* **Deployment:** Flask (Local or Cloud platforms like Render/Heroku)

---

## 🔧 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ai-news-recommendation-system.git](https://github.com/your-username/ai-news-recommendation-system.git)
    cd ai-news-recommendation-system
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    Access the application at `http://127.0.0.1:5000`.

---

## 🧠 Core Methodology

### 1. Content-Based Filtering
The system processes article text using **TfidfVectorizer** (removing English stop words). When a query is entered, it calculates the **Cosine Similarity** between the query vector and the pre-computed `tfidf_matrix` to find the most relevant articles based on textual context.

### 2. Collaborative Filtering
Using a synthetic user-item interaction matrix, the system fits a **Nearest Neighbors** model with a `cosine` metric. It identifies users with similar reading patterns and recommends articles those peers have rated highly that the current user hasn't interacted with yet.

---
##  Project Structure

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

