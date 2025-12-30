
# Sentiment-Based Product Recommendation System (Capstone)

## Project Overview

- **Goal**: Build a recommendation system that suggests products to users based on their sentiment expressed in reviews.
- **Approach**: 
  - Built a **Sentiment Analysis Model** using textual features from reviews.
  - Integrated **User-User Collaborative Filtering** for personalized recommendations.
  - Dockerized for deployments

---

## Key Highlights
In the highly competitive e-commerce landscape, providing personalized and accurate product recommendations is crucial for improving user experience and driving sales. Ebuss, a growing e-commerce company, aims to enhance its recommendation system by incorporating user sentiments derived from past reviews and ratings.

The goal of this project is to develop a Sentiment-Based Product Recommendation System that leverages Natural Language Processing (NLP) and machine learning techniques to analyze customer sentiments and refine product suggestions.

The project involves:
1. Lexical processing 

	Lowercasing
	Punctuation removal
	Stopwords removal
	Lemmatization
	Noise removal
2. Feature extraction

	TF- IDF model
	Bag of words model
	Count vectoriser
3. ML models for Sentiment Analysis

	Logistic regression
	Naive Bayes
	XGBoost
	Random forest
	Oversampling techniques
	Hyperparameter tuning
4. Recommendation system

	User-based recommendation system
	Item-based recommendation system
5. Deployment

	Flask
	Heroku


---

##  Tech Stack

- **Python**  
- **Pandas**, **NumPy**, **XGBoost**
- **FastAPI + Jinja2**
- **Docker**, **Uvicorn**
- **Railway (optional deployment)**

---

##  How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/gauravmehrotra-hsc/SentimentBasedProductRecommendation/
    cd Sentiment-Based-Product-Recommendation-Analysis
    ```

2. Install dependencies: (optional)
    ```bash
    pip install -r requirements.txt
    ```

3. Build and run the Docker container:
    ```bash
    docker build -t sentiment-recommendation-system .
    docker run -p 8000:8000 sentiment-recommendation-system
    ```

Then open `http://localhost:8000` in your browser.

---

##  Folder Structure

- `Sentiment_Recommendation_Capstone.ipynb`: Main notebook
- `app.py`: FastAPI app logic
- `model.py`: Sentiment + recommendation engine
- `templates/`: Contains Jinja2 HTML files
- `pickle/`: Trained models (excluded in repo)
- `Dockerfile`, `Procfile`, `requirements.txt`, `runtime.txt`: For Docker & Railway deployment

---

##  Results and Insights

- Predicts sentiment from user reviews and recommends relevant products.
- Containerized setup allows reproducible testing across platforms.


