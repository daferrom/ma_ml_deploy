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
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((256, 256))  # Tamaño típico para EfficientNet
        image = image.convert("RGB")      # Asegurar que es RGB
        image_array = np.array(image) / 255.0  # Normalización
        img_batch = np.expand_dims(image_array, axis=0)

        # Hacer la predicción
        prediction_probability = final_model.predict(img_batch)
        predicted_class_number = int(np.argmax(prediction_probability, axis=1)[0])

        return JSONResponse({"prediction": predicted_class_number})
    except Exception as e:
        return JSONResponse({"error": f"Hubo un problema procesando la imagen: {str(e)}"}, status_code=500)