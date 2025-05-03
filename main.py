import pickle

from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from dtos import *


app = FastAPI()

templates = Jinja2Templates(directory="templates")
templates.env.globals['url_for'] = app.url_path_for

with open("models/best_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
scaler = artifact["scaler"]
encoder = artifact["encoder"]

def get_phone_request(
        price_usd:    float = Form(..., gt=0),
        battery_mah:  float = Form(..., ge=0),
        ram_gb:       float = Form(..., gt=0),
        storage_gb:   float = Form(..., gt=0),
        camera_mp:    float = Form(..., gt=0),
        screen_size_in: float = Form(..., gt=0),
        weight_g:     float = Form(..., gt=0),
    ):
    return PhoneRequest(
        price=price_usd,
        battery=battery_mah,
        ram=ram_gb,
        storage=storage_gb,
        camera=camera_mp,
        screen=screen_size_in,
        weight=weight_g
    )

@app.get("/", response_class=HTMLResponse)
async def initialize(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
        request: Request,
        phone: PhoneRequest = Depends(get_phone_request)):

    data = phone.toArray()
    data_scaled = scaler.transform(data)


    pred_idx = model.predict(data_scaled)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": pred_label,
            "price_usd": phone.price,
            "ram_gb": phone.ram,
            "storage_gb": phone.storage,
            "battery_mah": phone.battery,
            "camera_mp": phone.camera,
            "screen_size_in": phone.screen,
            "weight_g": phone.weight
        }
    )

