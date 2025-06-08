from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model (placeholder)
model = tf.keras.models.load_model('models/matting_unet.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess and predict
        image = Image.open(filepath).resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        matte = model.predict(image_array)[0]
        matte = (matte * 255).astype(np.uint8)
        
        # Save matte
        matte_path = os.path.join(app.config['UPLOAD_FOLDER'], 'matte_' + file.filename.split('.')[0] + '.png')
        Image.fromarray(matte.squeeze()).save(matte_path)
        
        return render_template('index.html', 
                             result='Alpha matte generated',
                             image_url=filepath,
                             matte_url=matte_path)

if __name__ == '__main__':
    app.run(debug=True)