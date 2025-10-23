from fastapi import FastAPI, File, UploadFile 
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import io
app = FastAPI()
class_indexes_path = './models/dog_breed_25_class_indices.json'
MODEL_PATH = './models/dog_breed_25_classifier_model.h5'

print("Loading Keras model...") 

try:
        model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"!!! Error loading model: {e}")
    exit()
print("Model loaded successfully!") 

print("Loading class names...")
try:
    with open(class_indexes_path, 'r') as f:
        class_indices = json.load(f)
    # จัดเรียงชื่อคลาสตาม index (0, 1, 2, ...) เพื่อให้ตรงกับ output ของโมเดล
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    print(f"Loaded {len(class_names)} class names successfully!")
except FileNotFoundError:
    print(f"!!! Error: Class indices file not found at {class_indexes_path}")
    class_names = [] 
except Exception as e:
    print(f"!!! Error loading class names: {e}")
    class_names = []

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfile/" )
async def upload_file(file: UploadFile | None = None):
    if not file:
        return {"message ":"No file uploadede"}
    class_indexes_path
    try: 
        img = await file.read()
        img = Image.open(io.BytesIO(img))  #แปลงไบต์ให้เป็นรูปภาพ
        img = img.convert("RGB")
        img = img.resize((224,224)) # mobilenetV2 ต้องใช้ขนาดนี้เพราะ???
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
    except Exception as e:
         return {"message": "Error processing image", "error": str(e)}
    try:
        prediction = model.predict(img_batch)
        confidence = np.max(prediction[0]) * 100
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = class_names[predicted_class_index]
    except Exception as e:
        return {"message": "Error during prediction", "error": str(e)}

    if confidence < 70:
        predicted_class_name = "ไม่มั่นใจว่าเป็นพันธุ์อะไร" 

    return {
        "filename": file.filename,
        "predicted_breed": predicted_class_name,
        "confidence": f"{confidence:.2f}%"
    }