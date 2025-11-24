from flask import Flask, render_template, request, jsonify
from models.hepatitis_model import hepatitis_model
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_native_types(obj):
    """Convierte tipos numpy a tipos nativos de Python para JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

app = Flask(__name__)

try:
    hepatitis_model.load_model()
    hepatitis_model.load_scaler()
    logger.info("Modelo y Scaler cargados correctamente al iniciar la app")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnostic', methods=['GET'])
def diagnostic():
    try:
        test_data = {
            'Age': 45, 'Sex': 1, 'Estado_Civil': 1, 'Ciudad': 1, 'Steroid': 2,
            'Antivirals': 1, 'Fatigue': 2, 'Malaise': 1, 'Anorexia': 2, 
            'Liver_Big': 2, 'Liver_Firm': 1, 'Spleen_Palpable': 2, 'Spiders': 1,
            'Ascites': 2, 'Varices': 1, 'Bilirubin': 5.5, 'Alk_Phosphate': 180,
            'Sgot': 120, 'Albumin': 3.0, 'Protime': 35, 'Histology': 2
        }
        
        result = hepatitis_model.predict(test_data)
        result = convert_to_native_types(result)
        
        return jsonify({
            'model_status': 'loaded',
            'problem_detected': bool(hepatitis_model.problem_detected),
            'test_prediction': result,
            'features_count': len(hepatitis_model.features),
            'message': 'Modelo diagnosticado correctamente'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Recibiendo datos del formulario...")
        
        data = {}
        for feature in hepatitis_model.features:
            value = request.form.get(feature)
            if value is None or value == '':
                return render_template('index.html', 
                                     error=f"Campo faltante: {feature}",
                                     show_result=True)
            data[feature] = float(value)
        
        logger.info(f"Datos recibidos: {data}")
        result = hepatitis_model.predict(data)
        
        if 'error' in result:
            return render_template('index.html', 
                                 error=result['error'],
                                 show_result=True)
        
        logger.info(f"Resultado: {result}")
        
        return render_template('index.html', 
                             prediction=result['class'],
                             prediction_original=result['class_original'],
                             probability_positive=f"{result['probability_positive']*100:.2f}%",
                             probability_negative=f"{result['probability_negative']*100:.2f}%",
                             confidence=f"{result['confidence']*100:.2f}%",
                             problem_detected=result['problem_detected'],
                             message=result['message'],
                             show_result=True)
    
    except Exception as e:
        logger.error(f"Error en prediccion: {e}")
        return render_template('index.html', 
                             error=f"Error procesando la solicitud: {str(e)}",
                             show_result=True)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se recibieron datos JSON'}), 400
        
        logger.info(f"API Request: {data}")
        
        required_fields = hepatitis_model.features
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Campos faltantes: {missing_fields}'}), 400
        
        for key in data:
            try:
                data[key] = float(data[key])
            except (ValueError, TypeError):
                return jsonify({'error': f'Valor invalido para {key}: {data[key]}'}), 400
        
        result = hepatitis_model.predict(data)
        result = convert_to_native_types(result)
        
        logger.info(f"API Response: {result}")
        
        return jsonify({
            'status': 'success',
            'prediction': result['prediction'],
            'prediction_original': result['prediction_original'],
            'class': result['class'],
            'class_original': result['class_original'],
            'probability_positive': result['probability_positive'],
            'probability_negative': result['probability_negative'],
            'confidence': result['confidence'],
            'problem_detected': result['problem_detected'],
            'message': result['message'],
            'timestamp': '2024-01-01 00:00:00'
        })
    
    except Exception as e:
        logger.error(f"Error en API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK', 
        'message': 'API de Hepatitis funcionando',
        'model_loaded': hepatitis_model.model is not None,
        'scaler_loaded': hepatitis_model.scaler is not None,
        'problem_detected': bool(hepatitis_model.problem_detected)
    })

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        'features': hepatitis_model.features,
        'count': len(hepatitis_model.features)
    })

if __name__ == '__main__':
    logger.info("Iniciando servidor Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)