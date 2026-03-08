import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
import secrets
from PIL import Image
import numpy as np
import tensorflow as tf

# configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = secrets.token_hex(16)

model = None

# map output indices to human-readable labels
# TODO: replace these with the actual classes from your dataset
class_names = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy', 4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy', 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_', 9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot', 12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy', 15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy', 18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight', 21: 'Potato___Late_blight', 22: 'Potato___healthy', 23: 'Raspberry___healthy', 24: 'Soybean___healthy', 25: 'Squash___Powdery_mildew', 26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy', 28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight', 31: 'Tomato___Leaf_Mold', 32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite', 34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 36: 'Tomato___Tomato_mosaic_virus', 37: 'Tomato___healthy'}




def load_model():
    global model
    model = tf.keras.models.load_model('plant_disease_model.h5')
    print("Model loaded successfully")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(filepath):
    # load and preprocess image
    img = Image.open(filepath).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    # if you have a mapping of idx to class names, you can return name here
    return int(class_idx)


@app.route('/')
def upload():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        try:
            prediction = predict_image(filepath)

            # Save image to static folder for display
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(app.config.get('STATIC_FOLDER', 'static'), 'images', static_filename)
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            Image.open(filepath).save(static_path)

            # Store in session
            session['prediction'] = prediction
            session['image_path'] = f'images/{static_filename}'

            os.remove(filepath)  # Clean up temp file

            # redirect to result page for normal form submissions
            return redirect(url_for('result'))
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            # for form submissions, render upload with error message
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')

    if prediction is None:
        return redirect(url_for('upload'))

    label = class_names.get(prediction, f"Class {prediction}")
    return render_template('result.html', prediction=prediction, label=label, image_path=image_path)


if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
