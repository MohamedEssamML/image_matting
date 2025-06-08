# Image Matting

This project implements an image matting system to separate foreground objects from backgrounds using a simplified U-Net model. The system generates an alpha matte for input images. A Flask-based web interface allows users to upload images and view the resulting alpha matte.

## Features
- Generates alpha mattes using a U-Net-based model.
- Supports foreground-background separation.
- Web interface for uploading images and viewing results.
- Modular scripts for training and inference.

## Requirements
- Python 3.8+
- TensorFlow 2.5+
- Flask
- NumPy
- OpenCV
- Pillow
- See `requirements.txt` for a complete list.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MohamedEssamML/image_matting.git
   cd image_matting
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the `models/` directory contains the pretrained model (`matting_unet.h5`). Note: This is a placeholder; train the model or download pretrained weights.

## Usage
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open a browser and navigate to `http://localhost:5000`.
3. Use the web interface to upload an image and view the alpha matte.
4. Alternatively, run the prediction script directly:
   ```bash
   python predict_matting.py --image path/to/image.jpg
   ```
5. To train the model, prepare a dataset and run:
   ```bash
   python train_matting_model.py --dataset path/to/dataset
   ```

## Project Structure
```
image_matting/
├── app.py                    # Flask web application
├── train_matting_model.py    # Training script
├── predict_matting.py        # Inference script
├── models/                   # Pretrained model weights
├── static/                   # Static files (CSS, uploads)
├── templates/                # HTML templates
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── docs/                     # Detailed documentation
```

## Documentation
See `docs/documentation.md` for detailed information on the model architecture, dataset preparation, training, and deployment.

## Notes
- The pretrained model weight (`matting_unet.h5`) is a placeholder. Train the model using `train_matting_model.py` with a labeled dataset.
- Dataset should include images and corresponding alpha matte ground truths.
- This is a simplified implementation for educational purposes.
