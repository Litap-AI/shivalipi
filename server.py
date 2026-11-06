"""
Shivalipi Local Inference Server
Author: Rohit Manikrao Patil
"""

import base64
import io
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from shivalipi import Shivalipi

app = FastAPI(title="Shivalipi API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Shivalipi()


class TranscribeRequest(BaseModel):
    image_data: str


# Root route serving index.html directly
@app.get("/")
def serve_home():
    html_file = Path(__file__).parent / "index.html"
    return FileResponse(html_file)


@app.post("/transcribe")
def transcribe_doc(req: TranscribeRequest):
    data = req.image_data
    if "," in data:
        data = data.split(",")[1]

    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = engine.transcribe(image)
    return {"transcription": result.text, "raw": result.raw_text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
