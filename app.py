from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from model import product_recommendations_user  # Import recommendation function

# Initialize FastAPI app
app = FastAPI()

# Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# API Model
class UserInput(BaseModel):
    username: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "recommendations": None, "error_message": None})

@app.post("/", response_class=HTMLResponse)
async def get_recommendations(request: Request, user: str = Form(...)):
    recommendations = product_recommendations_user(user)

    if isinstance(recommendations, str):  # If function returns an error message
        error_message = recommendations
        recommendations = None
    elif recommendations.empty:  # If DataFrame is empty
        error_message = "No recommendations found."
        recommendations = None
    else:
        error_message = None

    return templates.TemplateResponse("index.html", {"request": request, "recommendations": recommendations, "error_message": error_message})

@app.post("/api/recommendations/")
async def api_recommendations(user: UserInput):
    recommendations = product_recommendations_user(user.username)

    if isinstance(recommendations, str):  # If an error message is returned
        return {"error": recommendations}

    return recommendations.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
