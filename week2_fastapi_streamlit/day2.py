from fastapi import FastAPI
from pydantic import BaseModel
import torch
from week2_fastapi_streamlit.model import CNN
import torch.nn.functional as F

app = FastAPI(title="MNIST Prediction API")

# Define request model
class PredictRequest(BaseModel):
    pixels: list  # flattened list of 784 pixels

# Load model once on startup
@app.on_event("startup")
def load_model():
    global model
    model = CNN()
    model.load_state_dict(torch.load("C:\\Users\\swastik dasgupta\\Desktop\\mlflow\\week1_pytorch\\models\\mnist_cnn.pth", map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")

# Root endpoint
@app.get("/")
def root():
    return {"message": "MNIST Prediction API is running!"}

# Predict endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    # Convert to tensor
    x = torch.tensor(request.pixels).view(1, 1, 28, 28).float()

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    return {"prediction": pred_class, "confidence": round(confidence, 4)}
