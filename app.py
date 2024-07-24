from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

API_URL = "https://api-inference.huggingface.co/models/numind/NuExtract-large"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

app = FastAPI()

class ExtractionRequest(BaseModel):
    inputs: str

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()

@app.post("/extract")
async def extract(data: ExtractionRequest):
    result = query(data.dict())
    return result

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Extraction API"}

