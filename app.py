from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import random

app = FastAPI()

# Mount the static folder (for CSS & JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_chat_page():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


