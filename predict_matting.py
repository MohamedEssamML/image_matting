import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import os

def predict_matte(model, image_path, output_path):
    """
    Predict an alpha matte for an input image using the U-Net model.
    
    Args:
        model: Loaded TensorFlow U-Net model.
        image_path (str): Path to the input image.
        output_path (str): Path to save the predicted alpha matte.
    
    Returns:
        str: Path where the alpha matte is saved.
    """
    # Load and preprocess image
    image = Image.open(image_path).resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict alpha matte
    matte = model.predict(image_array, verbose=0)[0]
    matte = (matte * 255).astype(np.uint8)
    
    # Save matte as PNG
    Image.fromarray(matte.squeeze()).save(output_path)
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict alpha matte for an image.')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    # Load the pretrained U-Net model
    model = tf.keras.models.load_model('models/matting_unet.h5')
    
    # Define output path
    output_filename = 'matte_' + os.path.basename(args.image).split('.')[0] + '.png'
    output_path = os.path.join('static/uploads', output_filename)
    
    # Ensure output directory exists
    os.makedirs('static/uploads', exist_ok=True)
    
    # Predict and save matte
    result_path = predict_matte(model, args.image, output_path)
    print(f'Alpha matte saved to {result_path}')