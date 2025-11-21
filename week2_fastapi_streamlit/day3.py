from fastapi import FastAPI
from pydantic import BaseModel
import torch
from week2_fastapi_streamlit.model import CNN
import torch.nn.functional as F
from fastapi import UploadFile, File
from PIL import Image
import numpy as np


app = FastAPI(title="MNIST Prediction API")

@app.on_event("startup")
def load_model():
    global model
    model=CNN()
    model.load_state_dict(torch.load("C:\\Users\\swastik dasgupta\\Desktop\\mlflow\\week1_pytorch\\models\\mnist_cnn.pth", map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")


# Root endpoint
@app.get("/")
def root():
    return {"message": "MNIST Prediction API is running!"}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    # 1. Read image
    image = Image.open(file.file).convert("L")   # convert to grayscale

    # 2. Resize to 28x28 (MNIST size)
    image = image.resize((28, 28))

    # 3. Convert to numpy
    img_array = np.array(image)

    # 4. Normalize pixel values
    img_array = img_array / 255.0

    # 5. Convert to tensor (1 batch, 1 channel, 28x28)
    x = torch.tensor(img_array).view(1, 1, 28, 28).float()

    # 6. Predict
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    return {
        "prediction": int(pred_class),
        "confidence": round(float(confidence), 4)
    }
