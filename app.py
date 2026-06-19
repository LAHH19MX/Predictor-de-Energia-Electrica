from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Cargar modelo y scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo y scaler cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        AT = float(request.form['AT'])
        V = float(request.form['V'])
        AP = float(request.form['AP'])
        RH = float(request.form['RH'])

        data_df = pd.DataFrame([[AT, V, AP, RH]], columns=['AT', 'V', 'AP', 'RH'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        data_scaled = scaler.transform(data_df)
        prediction = model.predict(data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        return jsonify({'prediccion': round(prediction[0], 2)})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)