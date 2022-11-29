from re import X
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import json
import torch

from transformers import BertTokenizer, BertForMaskedLM, BertConfig

output_model_file = "./BERTmodel.bin"
output_config_file = "./BERTconfig.bin"
output_vocab_file = "./BERTvocab.bin"

config = BertConfig.from_json_file(output_config_file)
model = BertForMaskedLM(config)
state_dict = torch.load(output_model_file)
model.load_state_dict(state_dict)
tokenizer = BertTokenizer(output_vocab_file)

app = FastAPI()

"""
objects = [{ "id": 1, "body": "hello"}, {"id": 2, "body": "world"}, {"id": 3, "body": "foo" }]

@app.get("/items/{item_id}}")
def fetch_item(item_id: int):
    for ob in objects:
        if ob['id'] == item_id :
            return {"id": item_id, "body": ob["body"]}
        
@app.post("/newitem")
def add_item(bodytxt: str):
    objects.append({"id": len(objects) + 1, "body": bodytxt})
    return {"id": objects[len(objects) - 1]["id"], "body": objects[len(objects) - 1]["body"]}
    # return {"id": len(objects), "body": bodytxt}
"""

@app.get("/")
async def root():
    string = "I went to the [MASK] today."
    inputs = tokenizer(string, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    return {"message": tokenizer.decode(predicted_token_id)}

"""
@app.get("/")
async def root():
    return {"message": "Hello World"}
"""