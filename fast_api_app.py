# fast_api_app.py

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import torch
from PIL import Image
import numpy as np
from models import get_model

app = FastAPI()

# Load the model
model = get_model("best_model").to("cpu")
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

def preprocess_image(file: UploadFile):
    # Convert uploaded file to grayscale image and preprocess
    image = Image.open(BytesIO(file.file.read())).convert("L")
    image = np.array(image) / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = preprocess_image(file)
    with torch.no_grad():
        prediction = model(image)
        prediction = torch.sigmoid(prediction).squeeze(0).squeeze(0).numpy()
    return {"prediction": prediction.tolist()}
