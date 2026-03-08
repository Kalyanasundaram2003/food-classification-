from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Download model from Google Drive if not exists
MODEL_PATH = 'best_model.h5'
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        'https://drive.google.com/uc?id=1g78hbXUQaaohEtJbfvA7huhKJz1TVeN4',
        MODEL_PATH, quiet=False
    )

model = load_model(MODEL_PATH)
nutrition_df = pd.read_csv('final_corrected_nutrition.csv')

class_names = ['badusha', 'chapati', 'cup_cakes', 'curd rice', 'dosa',
               'gulab jamun', 'french_fries', 'idly', 'jalaebi', 'kaju katli',
               'laddu', 'lemon rice', 'mysore pak', 'samosa', 'waffles']

CONFIDENCE_THRESHOLD = 0.6

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    max_prob = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    if max_prob < CONFIDENCE_THRESHOLD:
        predicted_class = "Not Found"
        nutrition_data = {}
    else:
        nutrition_row = nutrition_df[nutrition_df['food'].str.lower() == predicted_class.lower()]
        if not nutrition_row.empty:
            nutrition_data = nutrition_row.iloc[0].to_dict()
            nutrition_data.pop('food', None)
        else:
            nutrition_data = {}

    return render_template('result.html',
                           image_filename=filename,
                           prediction=predicted_class,
                           confidence=round(max_prob * 100, 2),
                           nutrition=nutrition_data)

if __name__ == '__main__':
    app.run(debug=False)
