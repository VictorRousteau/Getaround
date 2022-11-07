import pandas as pd

import mlflow
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("https://getaroundmlfowserver.herokuapp.com/")

description = """
# This is a simple API for the getaround.com pricing prediction project.
"""

tags_metadata = [

    {
        "name": "Introduction Endpoint",
        "description": "Getaround API"
    },
    {
        "name": "Predictor",
        "description": "Voting Regressor Experiment Predictor endpoint",
    }

]

app = FastAPI(
    title="GetAround pricing Experiment",
    description=description,
    version="0.1.0",
    openapi_tags=tags_metadata
)

###
# Here you define enpoints 
###
class PredictorFeatures(BaseModel):
    """
    Predictor Features
    """
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


@app.get("/", tags=["Introduction Endpoint"])
async def index():
    """
    API for Getaround
    """
    message = """
    Welcome to my API for getaround : prediction pricing project.
    """
    return message

@app.post("/predict", tags=["Predictor"])
async def predictor(features: PredictorFeatures):
    """
    Voting Regressor Experiment Predictor endpoint
    """
    # Read data
    input_df = pd.DataFrame(dict(features), index=[0])

    # Load model
    logged_model = 'runs:/a348d7b70a7649bb949e06b8613159a0/pricing-model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    prediction = loaded_model.predict(input_df)

    return {"prediction": prediction.tolist()[0]}


    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)