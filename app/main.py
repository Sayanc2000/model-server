from tkinter.tix import FileSelectBox
from fastapi import FastAPI, File, UploadFile
from typing import Optional

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/check/")
def check(file: Optional[UploadFile] = None):
    return {
        "file_name": file.filename,
        "file_content": file.file.read().decode("utf-8")
    }