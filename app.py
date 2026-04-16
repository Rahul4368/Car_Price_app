from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np

app = FastAPI()

# load model safely
model = pickle.load(open("car_price_model.pkl", "rb"))

templates = Jinja2Templates(directory="templates")

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None}
    )

# Predict route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    year: int = Form(...),
    mileage: float = Form(...),
    tax: float = Form(...),
    mpg: float = Form(...),
    engineSize: float = Form(...)
):
    try:
        features = np.array([[year, mileage, tax, mpg, engineSize]])

        pred = model.predict(features)

        # 🔥 super safe conversion
        if isinstance(pred, (list, tuple, np.ndarray)):
            prediction = float(pred[0])
        else:
            prediction = float(pred)

    except Exception as e:
        prediction = f"Error: {str(e)}"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": prediction}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=9000)