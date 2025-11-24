import joblib
import pandas as pd
import numpy as np

def diagnosticar_modelo():
    print("INICIANDO DIAGNOSTICO DEL MODELO...")
    
    try:
        modelo = joblib.load('modelo_regresion_logistica (4).pkl')
        print("Modelo cargado correctamente")
        
        scaler = joblib.load('scaler (1).pkl')
        print("Scaler cargado correctamente")
        
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        return
    
    print("\nINFORMACION DEL MODELO:")
    print(f"Clases: {modelo.classes_}")
    print(f"Coeficientes shape: {modelo.coef_.shape}")
    print(f"Intercepto: {modelo.intercept_}")
    
    print("\nPROBANDO CON DATOS EXTREMOS:")
    
    datos_normal = {
        'Age': 30, 'Sex': 1, 'Estado_Civil': 1, 'Ciudad': 1, 'Steroid': 1,
        'Antivirals': 1, 'Fatigue': 1, 'Malaise': 1, 'Anorexia': 1, 
        'Liver_Big': 1, 'Liver_Firm': 1, 'Spleen_Palpable': 1, 'Spiders': 1,
        'Ascites': 1, 'Varices': 1, 'Bilirubin': 1.0, 'Alk_Phosphate': 80,
        'Sgot': 35, 'Albumin': 4.5, 'Protime': 40, 'Histology': 1
    }
    
    datos_hepatitis = {
        'Age': 65, 'Sex': 1, 'Estado_Civil': 1, 'Ciudad': 1, 'Steroid': 2,
        'Antivirals': 2, 'Fatigue': 2, 'Malaise': 2, 'Anorexia': 2, 
        'Liver_Big': 2, 'Liver_Firm': 2, 'Spleen_Palpable': 2, 'Spiders': 2,
        'Ascites': 2, 'Varices': 2, 'Bilirubin': 8.5, 'Alk_Phosphate': 200,
        'Sgot': 150, 'Albumin': 2.0, 'Protime': 25, 'Histology': 2
    }
    
    for nombre, datos in [("NORMAL", datos_normal), ("HEPATITIS EXTREMA", datos_hepatitis)]:
        print(f"\n--- PRUEBA {nombre} ---")
        
        df = pd.DataFrame([datos])
        df = df[[
            'Age', 'Sex', 'Estado_Civil', 'Ciudad', 'Steroid', 'Antivirals',
            'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
            'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin',
            'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime', 'Histology'
        ]]
        
        datos_escalados = scaler.transform(df)
        prediccion = modelo.predict(datos_escalados)
        probabilidades = modelo.predict_proba(datos_escalados)
        
        print(f"Prediccion: {prediccion[0]}")
        print(f"Probabilidades: {probabilidades[0]}")
        print(f"Clase predicha: {'HEPATITIS' if prediccion[0] == 1 else 'NO HEPATITIS'}")
        
        if abs(probabilidades[0][0] - probabilidades[0][1]) > 0.8:
            print("ALERTA: Probabilidades muy sesgadas - posible desbalance")
    
    print("\nRECOMENDACIONES:")
    print("1. Si siempre predice lo mismo, ajustaremos el umbral de decision")
    print("2. Mostraremos las probabilidades en lugar de solo la clase")
    print("3. Implementaremos multiples estrategias de prediccion")

if __name__ == "__main__":
    diagnosticar_modelo()