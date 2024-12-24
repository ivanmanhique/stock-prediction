from fastapi import APIRouter, UploadFile, Depends
from sqlmodel import Session, select
from app.models import MlModel, get_session  # Adjust import path as needed
from app.services import continueTrain, predict

my_router = APIRouter()

@my_router.post('/continue-train')
async def continue_training(model_name:str, 
                            train_input: UploadFile,
                            new_model_name: str, session: Session = Depends(get_session)):
    
    metrics = continueTrain(model_name=model_name, train_input=train_input, newModelname=new_model_name)
    new_model = MlModel(name=new_model_name)
    # Add to the database
    session.add(new_model)
    session.commit()
    session.refresh(new_model)
    return metrics


@my_router.post('/predict')
async def preditct(model_name:str, input: UploadFile):
    predictions =  predict(model_name, input)
    return predictions


@my_router.get('/models', response_model=list[str])
async def getModels(session: Session = Depends(get_session)) -> list[str]:
    # Query the database for all models
    statement = select(MlModel.name)
    results = session.exec(statement).all()
    # Transform the results into a plain list of strings
    model_names = [result for result in results]

    # Return the list of model names
    return model_names