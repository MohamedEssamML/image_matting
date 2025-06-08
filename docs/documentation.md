# Image Matting Documentation

## Overview
This project implements an image matting system to separate foreground objects from backgrounds using a simplified U-Net model. The system generates an alpha matte to isolate the foreground. A Flask web interface allows users to upload images and view the resulting alpha matte.

## Model Architecture
- **Model**: Simplified U-Net
- **Input**: 256x256x3 RGB images
- **Architecture**:
  - **Encoder**: 4 blocks (Conv2D, BatchNormalization, ReLU, MaxPooling)
  - **Bottleneck**: Conv2D, BatchNormalization, ReLU
  - **Decoder**: 4 blocks (UpSampling, Conv2D, BatchNormalization, ReLU) with skip connections
  - **Output**: 256x256x1 alpha matte (sigmoid activation)
- **Loss**: Binary cross-entropy + L1 loss
- **Optimizer**: Adam (learning rate 1e-3)

## Dataset Preparation
- **Format**: Directory structure with images and alpha mattes:
  ```
  dataset/
  ├── images/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── mattes/
  │   ├── image1.png
  │   ├── image2.png
  │   └── ...
  ```
- **Preprocessing**:
  - Resize images and mattes to 256x256.
  - Normalize images to [0, 1].
  - Ensure mattes are grayscale with values [0, 1].
- **Recommended Datasets**: Adobe Deep Image Matting Dataset, Composition-1k.

## Training
1. Prepare a dataset with images and corresponding alpha mattes.
2. Run the training script:
   ```bash
   python train_matting_model.py --dataset path/to/dataset
   ```
3. The script:
   - Loads and preprocesses the dataset.
   - Trains the U-Net model.
   - Saves the trained model to `models/matting_unet.h5`.

## Inference
- **Script (`predict_matting.py`)**:
  - Loads the pretrained model.
  - Preprocesses the input image.
  - Outputs the alpha matte as an image.
- **Web Interface (`app.py`)**:
  - Upload an image via the Flask interface.
  - Displays the predicted alpha matte.

## Deployment
1. Install dependencies (`requirements.txt`).
2. Place the pretrained model in `models/`.
3. Run `app.py` to start the Flask server.
4. Access at `http://localhost:5000`.

## Implementation Details
- **Training (`train_matting_model.py`)**:
  - Uses TensorFlow's `tf.data` for efficient data loading.
  - Combines BCE and L1 losses for accurate matte prediction.
- **Inference (`predict_matting.py`)**:
  - Preprocesses images to match training conditions.
  - Saves the alpha matte as a PNG image.
- **Web Interface (`app.py`)**:
  - Flask application with routes for uploading images and displaying results.

## Future Improvements
- Use advanced matting models (e.g., DeepLabV3+, MODNet).
- Incorporate trimap inputs for better accuracy.
- Add support for video matting.
- Optimize for real-time performance.

## References
- Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- Xu, N., et al. (2017). Deep Image Matting.