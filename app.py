import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cohere

# Configure Cohere API Key (Replace with your actual key)
COHERE_API_KEY ="Hcci8rLMQbyjj8lfnknpQWwLs6tqYpCm9WFRtP1f"
co = cohere.Client(COHERE_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = 'E:/VScode/ASAP/Models/best_model.keras'
model = load_model(model_path)
print("Model input shape:", model.input_shape)

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

# Ensure 'uploads' folder exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

@app.route('/')
def index():
    return render_template('index.html')

def get_cohere_response(disease_name):
    """Get remedy suggestions from Cohere API for the detected disease."""
    try:
        prompt = f"What are the remedies for {disease_name} in plants?"
        response = co.generate(
            model="command",  # Use "command" instead of "command-r"
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
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image
    img_path = os.path.join('static/uploads', file.filename)
    file.save(img_path)

    # Preprocess the image
    img = Image.open(img_path).convert("RGB")  
    img = img.resize((150, 150))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    try:
        print("Model input shape:", model.input_shape)

        # Predict plant disease
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        # Get disease name
        predicted_class_name = class_labels[predicted_class]

        # Get remedy from Cohere API
        remedy = get_cohere_response(predicted_class_name)

        return render_template('result.html', 
                               predicted_class=predicted_class_name,
                               confidence=confidence * 100,
                               image_filename=file.filename,
                               remedy=remedy)

    except ValueError as e:
        return jsonify({
            'error': 'Model input shape mismatch. Ensure the model supports (150, 150, 3) input images.',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
