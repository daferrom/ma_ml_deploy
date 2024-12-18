from tensorflow.keras.models import load_model
import joblib

model_path = "best_efficientnet_model_V2.keras.h5"

def defineModel():
    model = load_model(model_path)
    print(model.summary())
    
    joblib.dump(model, "model.joblib")
    return

# Ejecutar la búsqueda de hiperparámetros
if __name__ == "__main__":
    defineModel()