from fastapi.responses import JSONResponse
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile


app = FastAPI()

final_model = tf.keras.models.load_model("best_efficientnet_model_V3.keras")

@app.post("/predict-land-use/")
async def predict_land_use(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    
    # Preprocesar la imagen según lo requiera EfficientNet
    # Comentado por ahora
    image = image.resize((224, 224))  # Tamaño típico para EfficientNet
    image = image.convert("RGB")     # Asegurar que es RGB
    image_array = np.array(image) / 255.0  # Normalización
    # Expandir las dimensiones de la imagen para simular un lote de tamaño 1
    
    # Expandir las dimensiones para simular un batch
    img_batch = np.expand_dims(image_array, axis=0)
    
    # Pasar la imagen al modelo (esto depende de cómo hayas configurado EfficientNet)
    prediction_probability = final_model.predict(img_batch)
    
    predicted_class_number = np.argmax(prediction_probability, axis=1)[0]  # Obtener la clase predicha
    
    return JSONResponse({"prediction": predicted_class_number})