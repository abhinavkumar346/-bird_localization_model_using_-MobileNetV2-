# Bird Bounding Box Regressor (MobileNetV2)

A transfer-learning model that localizes a single bird within an image by predicting its bounding box coordinates. Built on top of **MobileNetV2** (ImageNet pretrained) with a custom regression head, trained on the **Caltech Birds-2010** dataset.

## Overview

This project treats object localization as a regression problem: given an image containing one bird, the model directly predicts four values — `[xmin, ymin, xmax, ymax]` — representing the bounding box around the bird, normalized to the range [0, 1].

## Model Architecture

![Model architecture](results/architecture.png)

- **Backbone:** MobileNetV2 (`include_top=False`, ImageNet weights, fully fine-tuned)
- **Head:**
  - GlobalAveragePooling2D
  - Flatten
  - Dense(1024, activation='relu')
  - Dense(512, activation='relu')
  - Dense(4) — output layer for `[xmin, ymin, xmax, ymax]`
- **Input shape:** 224 x 224 x 3
- **Output:** Normalized bounding box coordinates `[xmin, ymin, xmax, ymax]` (range 0–1)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** SGD (momentum=0.9)

## Dataset

[Caltech Birds-2010 (CUB-200)](http://www.vision.caltech.edu/visipedia/CUB-200.html) — each image contains a single bird with one ground-truth bounding box.

## Training Details

- Epochs: 50
- Batch size: 32
- Train/validation split via TensorFlow Datasets (`caltech_birds2010`)

### Loss Curve

![Training and validation loss](results/loss_curve.png)

## Results

- **Validation loss (MSE):** _add your value here_
- **IoU evaluation:** Intersection-over-Union computed between predicted and ground-truth boxes on the validation set, with a threshold of 0.5 used to count "good" predictions.
  - Predictions with IoU ≥ 0.5: _add count_
  - Predictions with IoU < 0.5: _add count_

_Add a sample prediction image here (e.g. `results/sample_prediction.png`)._

## Usage

### Requirements

Create a `requirements.txt` with:

```
tensorflow>=2.18
keras>=3.9
tensorflow-datasets
numpy
matplotlib
opencv-python
pillow
```

Install with:

```bash
pip install -r requirements.txt
```


### Predict on a single image

```bash
python predict_bbox.py path/to/image.jpg --model birds.keras --output result.jpg
```

This will:
1. Load the trained model
2. Preprocess the image (resize to 224x224, normalise to [-1, 1])
3. Predict the normalised bounding box
4. Convert it to pixel coordinates for the original image
5. Draw the box on the image and save it

## Limitations

- The model assumes **exactly one bird per image** and always outputs one bounding box, even if no bird (or multiple birds) is present.
- No classification or confidence score is produced — it cannot indicate whether a bird is actually present.
- Performance is dataset-specific (Caltech Birds-2010); generalisation to other bird species, backgrounds, or live video may be limited.
- Not suitable for multi-object detection or real-time production use without further development (e.g., combining with a detection framework like YOLO or SSD).

## Acknowledgements

This project was built as part of the DeepLearning.AI / Coursera **TensorFlow: Advanced Techniques Specialisation** (Course 3, Week 1: Object Detection). The model architecture and training pipeline were implemented as a learning exercise in transfer learning and bounding box regression.

## License

_Add a license of your choice (e.g., MIT)._
