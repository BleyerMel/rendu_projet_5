from fastapi import FastAPI
import pandas as pd
import joblib
import json
import uvicorn
import cloudpickle
import mlflow

app = FastAPI()

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
logged_model = 'runs:/6c686c44df6543fca8617db1579e135b/model'
loaded_pipeline = mlflow.pyfunc.load_model(logged_model)


with open('labels.txt', 'r') as f:
    classes = [line.strip() for line in f]

# Inverser la transformation manuellement
def inverse_transform(encoded_labels, classes):
    return [classes[i] for i in encoded_labels]



@app.get('/')
def Alive():
    return 'Im Alive !'


@app.get('/predict')
def Prediction(data):
    
    # Création d'un df pour entrée de nouvelles données
    df = pd.DataFrame(eval(data))
    

    # Faire une prédiction avec la pipeline
    X_transformed = loaded_pipeline.predict(df)

    original_labels = inverse_transform(X_transformed, classes)

    return {"prediction": original_labels}

