from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.container import Container
from src.api.routes import router
from src.settings import Settings

load_dotenv()

settings = Settings()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator:
    container = Container()
    container.init_resources(settings)
    application.container = container  # type: ignore

    yield

    await container.cleanup()


app = FastAPI(
    title='RAG API for AI Course',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router)
