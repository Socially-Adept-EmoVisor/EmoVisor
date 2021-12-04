from fastapi import FastAPI

from .routes import router

app = FastAPI()

from . import startup  # noqa

app.include_router(router)
