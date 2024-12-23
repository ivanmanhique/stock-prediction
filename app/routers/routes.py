from fastapi import APIRouter, UploadFile, Depends
from sqlmodel import Session, select
from app.models import MlModel, get_session  # Adjust import path as needed
from app.services import continueTrain

my_router = APIRouter()

@my_router.post('/continue-train')
async def continue_training(model_name:str, train_input: UploadFile, new_model_name: str):
    metrics = continueTrain(model_name=model_name, train_input=train_input, newModelname=new_model_name)
    return metrics

@my_router.post('/predict')
async def predict(model_name:str, input: UploadFile):
    pass

@my_router.get('/models', response_model=list[str])
async def getModels(session: Session = Depends(get_session)) -> list[str]:
    # Query the database for all models
    statement = select(MlModel.name)
    results = session.exec(statement).all()

    # Return a list of model names
    return results