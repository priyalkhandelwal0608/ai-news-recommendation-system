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
- **Content-Based Recommendations**: Uses TF‑IDF vectorization and cosine similarity to recommend articles similar to a user’s query.  
- **Collaborative Filtering**: Suggests articles based on user–article interaction patterns.   
- **Professional Web Interface**: Flask backend with HTML/CSS frontend for recruiter‑friendly demos.  
- **Scalable Design**: Modular folder structure with reproducible pipelines.  

---
## Project Structure
ai-news-recommendation-system/
├── requirements.txt         # Python dependencies
├── app.py                   # Flask entry point (main web app)
│
├── data/                    # Datasets
│   ├── news_dataset.csv     # Synthetic or real news datase
├── src/                     # Core source code
│   ├── __init__.py
│   ├── recommender.py       # Content-based & collaborative filtering logic
│
│
├── templates/               # Frontend HTML 
│   └── index.html           # Main UI page
│
├── static/                  # Static assets (CSS)
│   ├── style.css
---

##  Tech Stack
- **Python** (Flask, Pandas, NumPy, scikit-learn)
- **NLP**: TF‑IDF vectorization
- **Frontend**: HTML, CSS
- **Deployment**: Flask (local or cloud platforms like Render/Heroku)

---


