from fastapi import FastAPI
from app.models import create_db_and_tables
from app.routers import my_router
from contextlib import asynccontextmanager



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run at startup
    create_db_and_tables()
    print("Database initialized successfully!")
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(my_router)
print("all is fine")