from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)

xgboostModel = joblib.load('models/xgboost_regression_model.joblib')
arimaModel = joblib.load('models/arima_model.joblib')

app = Flask(__name__)

@app.route('/forecast/regression', methods=['POST'])
def regressionForecast():
    try:
        data = request.json

        # Extract values 'v1' to 'v28' dynamically
        features = np.array([[float(data.get(f'V{i}')) for i in range(1, 29)]])
        
        # Make prediction
        prediction = xgboostModel.predict(features)[0]
        
        return jsonify({'forecast': float(prediction)})
    
    except Exception as e:
        return str(e), 400

@app.route('/forecast/timeseries', methods=['GET'])
def timeseriesForecast():
    try:
        steps = int(request.args.get('numberOfPredictions', 1))

        forecast = arimaModel.forecast(steps=steps)

        return jsonify({'forecasts': [float(f) for f in forecast]})
    
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='localhost',port=8001, debug=True)