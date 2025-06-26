from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# --- CONFIGURACIÓN CENTRAL DEL MODELO ---
MODEL_PATH = 'bean_area_predictor.joblib'
# Definimos explícitamente el orden de las características aquí.
# Este debe ser el mismo orden que se usó para entrenar el modelo.
EXPECTED_FEATURES = [
    'AspectRation', 
    'Compactness', 
    'ShapeFactor1', 
    'ShapeFactor2', 
    'ShapeFactor3', 
    'ShapeFactor4'
]
# Estándar de calidad
UMBRAL_CALIDAD = 50000

# --- Carga del modelo y los objetos guardados ---
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    scaler_X = model_bundle['scaler_X']
    scaler_Y = model_bundle['scaler_Y']
    print(f"✅ Modelo '{MODEL_PATH}' cargado exitosamente.")
except FileNotFoundError:
    print(f"❌ Error: El archivo del modelo '{MODEL_PATH}' no se encontró.")
    model = None
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model = None

# --- RUTA PRINCIPAL ---
@app.route('/')
def home():
    # Muestra el formulario web principal.
    return render_template('index.html')

# --- RUTA DE PREDICCIÓN ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', texto_resultado='Error: El modelo no está cargado.', texto_verificacion='')

    try:
        # --- PASO DE DEPURACIÓN Y LECTURA DE DATOS ---
        # Leemos los datos del formulario explícitamente por su nombre y en el orden correcto.
        input_data = [float(request.form[name]) for name in EXPECTED_FEATURES]
        
        # Imprimimos en la terminal para verificar que todo es correcto.
        print("\n--- INICIANDO PREDICCIÓN ---")
        print(f"Datos recibidos del formulario: {request.form}")
        print(f"Datos ordenados para el modelo: {input_data}")

        # Crear un DataFrame con los datos ya en el orden correcto
        input_df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
        
        # --- PROCESO DE PREDICCIÓN ---
        # 1. Escalar los datos de entrada
        input_scaled = scaler_X.transform(input_df)
        
        # 2. Realizar la predicción
        prediction_scaled = model.predict(input_scaled)
        
        # 3. Invertir el escalado del resultado para obtener el valor real
        prediction_orig = scaler_Y.inverse_transform(prediction_scaled.reshape(-1, 1))
        area_predicha = prediction_orig[0][0]
        
        print(f"Predicción (escalada): {prediction_scaled[0]:.4f} -> Predicción (original): {area_predicha:.2f}")

        # --- LÓGICA DE VERIFICACIÓN DE CALIDAD ---
        if area_predicha >= UMBRAL_CALIDAD:
            verificacion = f"<b>Estado: CUMPLE</b> con el estándar de calidad (> {UMBRAL_CALIDAD} pixeles²)."
        else:
            verificacion = f"<b>Estado: NO CUMPLE</b> con el estándar de calidad (> {UMBRAL_CALIDAD} pixeles²)."
        
        resultado = f"El área estimada (predicción de regresión) es: <strong>{area_predicha:.2f}</strong> pixeles²."
        print("--- PREDICCIÓN COMPLETADA ---")
        
    except Exception as e:
        resultado = f"❌ Error al procesar la entrada: {e}"
        verificacion = ""
        print(f"--- ERROR: {e} ---")

    return render_template('index.html', texto_resultado=resultado, texto_verificacion=verificacion)

# Ejecutar la aplicación
if __name__ == '__main__':
    # Usamos host='0.0.0.0' para que sea accesible en la red local
    app.run(host='0.0.0.0', port=5000, debug=True)

