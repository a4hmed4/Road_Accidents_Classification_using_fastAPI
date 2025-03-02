import pickle
import numpy as np
import os
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,  "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR,  "label_encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load the trained model, label encoders, and scaler
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoders = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Define categorical columns
categorical_columns = [
    "1st Road Class", "Road Surface", "Lighting Conditions",
    "Weather Conditions", "Casualty Severity", "Sex of Casualty",
    "Type of Vehicle", "age_group", "vehicle_group"
]

# Initialize FastAPI
app = FastAPI()

# Mount static files (for CSS, images)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "error": None})

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    num_vehicles: int = Form(...),
    time: str = Form(...),
    road_class: str = Form(...),
    road_surface: str = Form(...),
    lighting: str = Form(...),
    weather: str = Form(...),
    casualty_severity: str = Form(...),
    sex_of_casualty: str = Form(...),
    age_of_casualty: int = Form(...),
    type_of_vehicle: str = Form(...),
    age_group: str = Form(...),
    vehicle_group: str = Form(...)
):
    try:
        # Prepare input data
        form_data = {
            "Number of Vehicles": num_vehicles,
            "Time (24hr)": time,
            "1st Road Class": road_class,
            "Road Surface": road_surface,
            "Lighting Conditions": lighting,
            "Weather Conditions": weather,
            "Casualty Severity": casualty_severity,
            "Sex of Casualty": sex_of_casualty,
            "Age of Casualty": age_of_casualty,
            "Type of Vehicle": type_of_vehicle,
            "age_group": age_group,
            "vehicle_group": vehicle_group
        }

        # Encode categorical data
        encoded_input = []
        for col, value in form_data.items():
            if col in categorical_columns:
                encoded_value = label_encoders[col].transform([value])[0]
            else:
                encoded_value = value
            encoded_input.append(encoded_value)

        # Scale input
        input_array = np.array(encoded_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        predicted_class_name = label_encoders["Casualty Severity"].inverse_transform([prediction])[0]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": predicted_class_name,
            "error": None
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "error": str(e)
        })
