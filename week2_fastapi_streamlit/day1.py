from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()

# Root route
@app.get("/")
def root():
    return {"message": "🚀 FastAPI is running successfully!"}

# Health check route
@app.get("/health")
def health_check():
    return {"status": "ok", "server": "FastAPI", "ready": True}
