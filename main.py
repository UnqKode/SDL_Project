from fastapi import FastAPI
from pydantic import BaseModel,field_validator
import torch
import torch.nn.functional as F
import numpy as np
from tests.inputTest import validate_dim,validate_shape
from utils.load_model import load_saved_model
from utils.load_global_prototypes import load_global_prototypes
from utils.classify_queries import classify_queries
app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys, os
sys.path.append(os.path.dirname(__file__))


# expects data in {"input" : [[...],[...],....,[...]]} (6,128)


class HARSinput(BaseModel):
    input:list

    @field_validator("input")
    @classmethod
    def validate_input(cls,input):
        inp_arr = np.array(input)
        validate_shape(inp_arr)
        validate_dim(inp_arr)
        return input


prototypes,_ = load_global_prototypes('./data/global_prototypes.pth')

print("loading complete")

@app.post("/predict")
def predict_activity(data:HARSinput):
    input_np = np.array(data.input, dtype=np.float32)
    input_np = input_np.T
    input_t = torch.tensor(input_np).unsqueeze(0).to(device)
    model = load_saved_model('./data/protonet_har_model.pth',input_np)
    model.eval()
    with torch.no_grad():
        emb = model(input_t)
    
    logits = classify_queries(prototypes,emb)
    preds = torch.argmax(logits, dim=1)
    activity_labels = {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING",
    }
    probs = F.softmax(logits, dim=1)
    confidence = float(probs[0, preds].item())
    
    return {"predicted_class_index":preds,
            "predicted_class_name":activity_labels[int(preds.item())],
            "confidence":confidence
        }

print("loaded")