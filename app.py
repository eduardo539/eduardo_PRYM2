from flask import Flask, request, render_template, jsonify
import pandas as pd
import logging
import joblib
#from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo de regresión y el escalador
model_regressor = joblib.load('modelo_random_forest.pkl')
app.logger.debug('Modelo de regresión y escalador cargados correctamente.')

#redes_neuronales_TF = load_model('model2.h5')  # Cargar el modelo desde el archivo model.h5

# Cargar el modelo de redes neuronales
#redes_neuronales_TF = joblib.load('redes_neuronales_TF.pkl')
app.logger.debug('Modelo de redes neuronales cargado correctamente.')

# Crear el objeto StandardScaler
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        selected_model = request.form['selected_model']
        model_name = float(request.form['model_name'])
        lowest_price = float(request.form['lowest_price'])
        highest_price = float(request.form['highest_price'])
        screen_size = float(request.form['screen_size'])
        memory_size = float(request.form['memory_size'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[model_name, lowest_price, highest_price, screen_size, memory_size]], 
                               columns=['model_name', 'lowest_price', 'highest_price', 'screen_size', 'memory_size'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Seleccionar y usar el modelo adecuado
        if selected_model == 'regression':
            # Realizar predicciones con el modelo de regresión
            prediction = model_regressor.predict(data_df)
        elif selected_model == 'neural_network':
            # Escalar los datos usando el escalador previamente guardado
            #scaled_data = scaler.fit_transform(data_df)
            # Realizar predicciones con el modelo de redes neuronales
            #prediction = redes_neuronales_TF.predict(scaled_data)
            raise ValueError("Modelo redes neuronales")
        else:
            raise ValueError("Modelo no reconocido")
        
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'Mejor_precio': float(prediction[0])})
    except Exception as e:
        app.logger.error(f'Error en el procesamiento: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
