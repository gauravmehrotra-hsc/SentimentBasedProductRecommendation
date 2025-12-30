
# Docker Notes – Sentiment-Based Product Recommendation System

This document details the Docker-based deployment pipeline built for this project. 

---

## Folder Structure

Here’s how the project is structured to support containerized deployment:

```
├── app.py                  # FastAPI app with web + API routes
├── model.py                # Recommendation + sentiment logic
├── Dockerfile              # Docker image definition
├── Procfile / runtime.txt  # For Railway/Heroku deployment
├── templates/              # HTML UI (Jinja2), contains index.html
│   └── index.html
├── pickle/                 # Pre-trained models + vectorizers
```

---

## What the Docker Image Contains

The image created from this project includes:

- **Python 3.12 base**
- **FastAPI + Uvicorn** server
- Your full application code
- Mounted `templates/` and `pickle/` folders

---

## ⚙️ Build & Run Commands

To run the app locally using Docker:

```bash
# Step 1: Clone the repository
git clone https://github.com/..
cd SentimentBasedProductRecommendation

# Step 2: Build the image
docker build -t sentiment-recommendation-system .

# Step 3: Run the container
docker run -p 8000:8000 sentiment-recommendation-system
```

> The app will be accessible at: `http://localhost:8000`

---

## Notes on Model Files

The application depends on multiple `.pkl` files stored inside the `pickle/` directory:

| Filename | Description |
|----------|-------------|
| `user_final_rating.pkl` | Matrix of user-product ratings |
| `cleaned-data.pkl`      | Final cleaned product reviews |
| `tfidf-vectorizer.pkl`  | TF-IDF model for review text |
| `sentiment-classification-xg-boost-best-tuned.pkl` | Trained XGBoost classifier |

To use them:

---

## Deployment

- Designed for both **local use** and **Railway/Heroku** cloud deployment.
- `Procfile` and `runtime.txt` included to enable smooth deployment.

---


---
