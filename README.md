# AI News Recommendation System

##  Overview
The **AI News Recommendation System** is a machine learning application that provides personalized news article recommendations using **content-based filtering** and **collaborative filtering** techniques. It is designed with a professional web interface (Flask + HTML/CSS) and modular backend code, making it recruiter‑ready and production‑oriented.

---

##  Problem Statement
With the rapid growth of digital media, readers are overwhelmed by the sheer volume of news articles published daily. Traditional keyword search or static categorization fails to deliver personalized experiences.  
This project addresses the challenge by building a **data-driven recommender system** that can:
- Suggest relevant articles based on semantic similarity of content.
- Recommend articles by learning from user–article interactions.
- Provide a hybrid approach combining both methods for improved accuracy.

---

##  Features

* **Content-Based Recommendations**: Uses **TF-IDF Vectorization** and **Cosine Similarity** to suggest articles based on the textual content of a search query.
* **Collaborative Filtering**: Implements a **K-Nearest Neighbors (KNN)** model to suggest articles liked by similar users based on interaction history.
* **Hybrid Logic**: Combines multiple strategies to overcome the "cold start" problem and improve discovery.
* **Web Dashboard**: A simple, responsive UI for interacting with the recommendation engine.


---

##  Project Structure

```text
AI-NEWS-RECOMMENDATION-SYSTEM/
├── data/
│   └── news_dataset.csv       # Dataset with article titles and content
├── src/
│   ├── recommender.py         # Logic for TF-IDF and KNN models
├── static/
│   └── style.css              # Custom styling for the web UI
├── templates/
│   └── index.html             # Flask template for the UI
├── app.py                     # Flask application entry point
└── requirements.txt           # Python dependencies


##  Tech Stack
- **Python** (Flask, Pandas, NumPy, scikit-learn)
- **NLP**: TF‑IDF vectorization
- **Frontend**: HTML, CSS
- **Deployment**: Flask (local or cloud platforms like Render/Heroku)



## Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/your-username/ai-news-recommendation-system.git](https://github.com/your-username/ai-news-recommendation-system.git)
cd ai-news-recommendation-system
Set up a virtual environment:

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the app:

Bash
python app.py
Access the application at http://127.0.0.1:5000.

Core Methodology
Content-Based Filtering
The system processes article text using TfidfVectorizer (removing English stop words). When a query is entered, it calculates the Cosine Similarity between the query vector and the pre-computed tfidf_matrix to find the most relevant articles.

Collaborative Filtering
Using a synthetic user-item interaction matrix, the system fits a Nearest Neighbors model with a cosine metric. It identifies users with similar reading patterns and recommends articles those peers have rated highly.


