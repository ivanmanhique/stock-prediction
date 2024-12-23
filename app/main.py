from fastapi import FastAPI
from .routers import my_router
from .models import create_db_and_tables
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run at startup
    create_db_and_tables()
    print("Database initialized successfully!")
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(my_router)