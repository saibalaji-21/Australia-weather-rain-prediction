import joblib
from flask import Flask,render_template,request
import numpy as np
import mlflow
import os

# MLflow + DagsHub setup
mlflow.set_tracking_uri("https://dagshub.com/saibalaji-21/Australia-weather-rain-prediction.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "saibalaji-21"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a5b420c5840a5c9eda2340c4632291eefefa6324"

app = Flask(__name__)

MODEL_PATH = "artifacts/models/model.pkl"
model = joblib.load(MODEL_PATH)


FEATURES = [
    'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'Year', 'Month', 'Day'
]

LABELS = {0 : "NO" , 1: "YES"}

@app.route("/" , methods=["GET" , "POST"])
def index():
    prediction = None

    if request.method=="POST":
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_array = np.array(input_data).reshape(1,-1)

            pred = model.predict(input_array)[0]
            prediction = LABELS.get(pred, 'Unknown')
            print(prediction)

            # MLflow logging
            with mlflow.start_run():
                for feature in FEATURES:
                    mlflow.log_param(feature, request.form.get(feature))
                mlflow.log_metric("Prediction", int(pred))

        except Exception as e:
            print(str(e))
    
    return render_template("index.html" , prediction=prediction , features=FEATURES)

if __name__=="__main__":
    app.run(debug=True , port=5000 , host="0.0.0.0")