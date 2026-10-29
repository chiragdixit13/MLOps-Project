import mlflow
import mlflow.sklearn
import fastapi
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI(title="Diabetes prediction API")


mlflow.set_tracking_uri(
     "sqlite:///C:/Users/DELL/Documents/AIML_roadmap/MLOps/MLFLOW/mlflow.db"
)

model=mlflow.sklearn.load_model(
    "models:/Diabetes-Prediction-Model/1"
)


class PatientData(BaseModel):
    age:float
    sex:float
    bmi:float
    bp:float
    s1:float
    s2:float
    s3:float
    s4:float
    s5:float
    s6:float

@app.get("/check")
def check():
    return {
        "message":"Diabetes Prediction API is working"
    }


@app.post("/predict")

def predict(request:PatientData):
    features=[[request.age,request.sex,request.bmi,request.bp,request.s1,request.s2,request.s3,request.s4,request.s5,request.s6]]
    prediction=model.predict(features)[0]
    return{"prediction":round(float(prediction),2)

    }
    