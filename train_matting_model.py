import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, BatchNormalization, ReLU
import numpy as np
from PIL import Image
import os
import argparse

def build_unet():
    inputs = Input((256, 256, 3))
    
    # Encoder
    c1 = Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(256, (3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)
    
    # Decoder
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, (3, 3), padding='same')(u4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)
    
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, (3, 3), padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = ReLU()(c5)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
    return model

def load_data(dataset_path):
    images = []
    mattes = []
    
    image_dir = os.path.join(dataset_path, 'images')
    matte_dir = os.path.join(dataset_path, 'mattes')
    
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        matte_path = os.path.join(matte_dir, filename.split('.')[0] + '.png')
        
        if os.path.exists(matte_path):
            image = Image.open(img_path).resize((256, 256))
            matte = Image.open(matte_image).resize((256,256))
            image = np.array(image) / 255.0
            matte = np.array(matte) / 255.0
            if len(matte.shape) == 3:
                matte = matte[:, :, 0]
            images.append(image)
            mattes.append(matte)
    
    return np.array(images), np.array(mattes)

def train(dataset_path):
    # Load data
    X, y = load_data(dataset_path)
    
    # Build and train model
    model = build_unet()
    model.fit(X, y, batch_size=16, epochs=20, validation_split=0.2, verbose=1)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/matting_unet.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    args = parser.parse_args()
    train(args.dataset)
```

5. **Prediction Script (predict_matting.py)**
<xaiArtifact artifact_id="a9f74e83-8edb-4881-b077-7296c0d63ec7" artifact_version_id="24b569aa-4d10-465f-86cb-9d3b110b83dc" title="predict_matting.py" contentType="text/python">
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import os

def predict_matte(model, image_path, output_path):
    # Load and preprocess image
    image = Image.open(image_path).resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    matte = model.predict(image_array)[0]
    matte = (matte * 255).astype(np.uint8)
    
    # Save matte
    Image.fromarray(matte.squeeze()).save(output_path)
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    model = tf.keras.models.load_model('models/matting_unet.h5')
    output_path = os.path.join('static', 'matte_' + os.path.basename(args.image).split('.')[0] + '.png')
    predict_matte(model, args.image, output_path)
    print(f'Matte saved to {output_path}')