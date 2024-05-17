import torch
import torch.nn as nn
from torch.nn import functional as F
import model
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

myModel = model.SimpleTransformerLanguageModel()
myModel.load_state_dict(torch.load('CityNameGenVer1.pth'))
myModel.eval()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (replace with specific origins if needed)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow POST and OPTIONS methods
    allow_headers=["*"],  # Allow all headers (replace with specific headers if needed)
)

class CreateData(BaseModel):
    count: int

class FinishData(BaseModel):
    features: str

@app.post("/create/")
async def create(input_data: CreateData):
    features = torch.tensor([[0]], dtype=torch.long)
    with torch.no_grad():
        output = myModel.generate(features, input_data.count)
    output = output[0].tolist()
    output.pop(0)
    return model.decode(output)

@app.post("/finish/")
async def finish(input_data: FinishData):
    encoded = model.encode(input_data.features)
    features = torch.tensor([encoded,], dtype=torch.long)
    with torch.no_grad():
        output = myModel.generate(features, 1)
    output = output[0].tolist()
    output.pop(0)
    return input_data.features + model.decode(output)

if __name__ == "__main__":
   uvicorn.run("App:app", host="127.0.0.1", port=8000, reload=True)
