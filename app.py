from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.debug = True

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/seed_model.h5'
IMG_SIZE = 128

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SEED_INFO = {
    'apple': {
        'description': 'Apple seeds are small, dark brown, and tear-shaped.',
        'origin': 'Central Asia',
        'germination_time': '30-90 days (needs cold stratification)',
        'nutrition': 'Contains amygdalin (cyanide compound) - not meant for nutrition',
        'climate': 'Temperate to cold climates',
        'planting_tips': 'Plant in well-drained soil; requires cross-pollination.',
        'pros': ['High in fiber and antioxidants', 'Good for heart health', 'May lower risk of diabetes'],
        'cons': ['Can cause bloating in some people', 'Seeds contain trace amounts of cyanide (if chewed)', 'Pesticide residue on skin if not organic']
    },
    'avocado': {
        'description': 'Avocado seeds are large, round or slightly oval, and brown.',
        'origin': 'South-Central Mexico',
        'germination_time': '2-6 weeks',
        'nutrition': 'High in antioxidants and potassium (extracts)',
        'climate': 'Tropical and subtropical',
        'planting_tips': 'Suspend seed in water using toothpicks until roots appear.',
        'pros': ['Rich in healthy monounsaturated fats', 'Excellent source of potassium', 'High in fiber'],
        'cons': ['High in calories', 'Can be expensive', 'Very short window of perfect ripeness']
    },
    'mango': {
        'description': 'Mango seeds are large, flat, and usually have a hard, fibrous outer shell.',
        'origin': 'South Asia',
        'germination_time': '2-4 weeks',
        'nutrition': 'Rich in fatty acids and minerals',
        'climate': 'Tropical and frost-free subtropical',
        'planting_tips': 'Remove hard husk before planting in warm, moist soil.',
        'pros': ['High in Vitamin C and A', 'Supports immune system', 'Contains digestive enzymes'],
        'cons': ['High in natural sugar', 'Skin can cause allergic reactions in some', 'Messy to eat']
    },
    'papaya': {
        'description': 'Papaya seeds are small, black, and round with a slightly gelatinous coating.',
        'origin': 'Central America and Southern Mexico',
        'germination_time': '2-4 weeks',
        'nutrition': 'High in protein, fats, and papain enzyme',
        'climate': 'Tropical (frost sensitive)',
        'planting_tips': 'Wash off gelatinous coating before planting in sunny spot.',
        'pros': ['Excellent for digestion (contains papain)', 'Rich in Vitamin C', 'Good for skin health'],
        'cons': ['Strong smell that some dislike', 'Can cause digestive issues if eaten unripe', 'Seeds are edible but have a strong peppery taste']
    },
    'watermelon': {
        'description': 'Watermelon seeds are small, flat, and can be black or pale/white.',
        'origin': 'Southern Africa',
        'germination_time': '4-10 days',
        'nutrition': 'Rich in magnesium, zinc, and iron',
        'climate': 'Warm to hot climates',
        'planting_tips': 'Plant in mounds of well-draining, rich soil with full sun.',
        'pros': ['Excellent for hydration (92% water)', 'Rich in lycopene', 'Low in calories'],
        'cons': ['High glycemic index', 'Can cause digestive distress in large amounts', 'Takes up a lot of fridge space']
    }
}

model = None
keras_image = None

# Only load the model in the active worker process to avoid loading it twice during Flask reload
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    print("Loading TensorFlow and seed classification model... (this may take a few seconds)", flush=True)
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image as temp_image
        keras_image = temp_image
        model = load_model(MODEL_PATH)
        print("Model loaded successfully", flush=True)
    except Exception as e:
        model = None
        keras_image = None
        print(f"Error loading model: {e}", flush=True)
else:
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = keras_image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index]) * 100
        
        class_names = sorted(os.listdir('dataset'))
        predicted_class = class_names[predicted_class_index]
        
        all_probabilities = []
        for i, class_name in enumerate(class_names):
            all_probabilities.append({
                'name': class_name,
                'probability': float(predictions[0][i]) * 100
            })
            
        all_probabilities = sorted(all_probabilities, key=lambda x: x['probability'], reverse=True)
        
        info = SEED_INFO.get(predicted_class, {'pros': [], 'cons': [], 'description': ''})
        
        filename = f"uploaded_{os.urandom(8).hex()}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        img.save(filepath)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'description': info.get('description', ''),
            'origin': info.get('origin', ''),
            'germination_time': info.get('germination_time', ''),
            'nutrition': info.get('nutrition', ''),
            'climate': info.get('climate', ''),
            'planting_tips': info.get('planting_tips', ''),
            'all_probabilities': all_probabilities,
            'image_path': f"/static/uploads/{filename}",
            'pros': info.get('pros', []),
            'cons': info.get('cons', [])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
