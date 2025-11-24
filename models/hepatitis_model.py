import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class HepatitisModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'Age', 'Sex', 'Estado_Civil', 'Ciudad', 'Steroid', 'Antivirals',
            'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
            'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin',
            'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime', 'Histology'
        ]
        self.problem_detected = False
        
    def load_model(self):
        try:
            self.model = joblib.load('modelo_regresion_logistica (4).pkl')
            print("Modelo cargado correctamente")
            self._diagnose_model()
            
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            raise e
    
    def load_scaler(self):
        try:
            self.scaler = joblib.load('scaler (1).pkl')
            print("Scaler cargado correctamente")
        except Exception as e:
            print(f"Error cargando el scaler: {e}")
            raise e
    
    def _diagnose_model(self):
        print("Diagnosticando modelo...")
        test_data = np.random.randn(10, len(self.features))
        predictions = self.model.predict(test_data)
        unique_predictions = np.unique(predictions)
        
        if len(unique_predictions) == 1:
            print(f"ALERTA: El modelo siempre predice: {unique_predictions[0]}")
            self.problem_detected = True
        else:
            print("El modelo hace predicciones variadas")
    
    def predict(self, input_data):
        try:
            df = pd.DataFrame([input_data])
            df = df[self.features]
            scaled_data = self.scaler.transform(df)
            
            prediction_normal = self.model.predict(scaled_data)[0]
            probabilities = self.model.predict_proba(scaled_data)[0]
            
            if self.problem_detected:
                print("Aplicando estrategia para modelo problematico...")
                prob_positive = probabilities[1]
                
                if prob_positive > 0.3:
                    final_prediction = 1
                else:
                    final_prediction = 0
            else:
                final_prediction = prediction_normal
            
            confidence = max(probabilities)
            
            return {
                'prediction': final_prediction,
                'prediction_original': prediction_normal,
                'probability_negative': float(probabilities[0]),
                'probability_positive': float(probabilities[1]),
                'confidence': float(confidence),
                'class': 'HEPATITIS' if final_prediction == 1 else 'NO HEPATITIS',
                'class_original': 'HEPATITIS' if prediction_normal == 1 else 'NO HEPATITIS',
                'problem_detected': self.problem_detected,
                'message': 'Modelo con posible desbalance - verificar probabilidades' 
                          if self.problem_detected else 'Prediccion normal'
            }
            
        except Exception as e:
            print(f"Error en prediccion: {e}")
            return {
                'error': str(e),
                'prediction': 0,
                'class': 'ERROR',
                'probability_negative': 0.5,
                'probability_positive': 0.5,
                'confidence': 0
            }

hepatitis_model = HepatitisModel()