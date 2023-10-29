from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import pdf

app = FastAPI()
app.include_router(pdf.router, prefix="/resume/pdf")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def status():
    return {
        "status": "Let's parse CV...",
        "timestamp": datetime.timestamp(datetime.now()),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=8080, host="0.0.0.0")
