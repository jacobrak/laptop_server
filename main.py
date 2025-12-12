from fastapi import FastAPI
from pydantic import BaseModel
from model import *
# ---- request body ----
class TextRequest(BaseModel):
    text: str

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello: World"}

@app.post("/uwu")
def uwu(req: TextRequest):
    return {
        "uwu": encode(req.text)
    }