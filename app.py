import os
import requests
import gdown
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cohere

# # Configure Cohere API Key (Set this as an environment variable for security)
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# co = cohere.Client(COHERE_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Google Drive File ID for the model
FILE_ID = "13yrVVV-wXBM5q3rZdPAWfGROh6hMWoXx"
MODEL_PATH = "static/models/best_model.keras"

# Ensure necessary folders exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/models", exist_ok=True)

def download_model():
    """Download model from Google Drive if not exists."""
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists, skipping download.")

# Download and load model
download_model()
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Class labels for plant diseases
class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

@app.route('/')
def index():
    return render_template('index.html')

def get_cohere_response(disease_name):
    """Get remedy suggestions from Cohere API for the detected disease."""
    if not COHERE_API_KEY:
        return "Cohere API key is missing. Set it as an environment variable."
    
    try:
        prompt = f"What are the remedies for {disease_name} in plants?"
        response = co.generate(
            model="command",  
            prompt=prompt,
            max_tokens=100
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error fetching remedy: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    """Handles plant disease prediction and provides remedies."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    img_path = os.path.join("static/uploads", file.filename)
    file.save(img_path)

    # Preprocess the image
    img = Image.open(img_path).convert("RGB")  
    img = img.resize((150, 150))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    try:
        # Predict plant disease
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        # Get disease name
        predicted_class_name = class_labels.get(predicted_class, "Unknown")

        # Get remedy from Cohere API
        remedy = get_cohere_response(predicted_class_name)

        return render_template('result.html', 
                               predicted_class=predicted_class_name,
                               confidence=confidence * 100,
                               image_filename=file.filename,
                               remedy=remedy)

    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
