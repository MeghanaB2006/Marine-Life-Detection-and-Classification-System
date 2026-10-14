# Closed-Loop Marine Monitoring System Using Multi-Dataset Deep Learning

## 1. Project Overview

The **Closed-Loop Marine Monitoring System** is a computer vision-based application designed to detect marine organisms and classify fish species from underwater and pond images.

The system combines deep learning models for image enhancement, object detection, and species classification with a rule-based feedback controller. The controller evaluates detection quality and enables additional detection attempts when the initial results do not meet predefined criteria.

The project aims to support automated fish monitoring, species identification, and analysis of marine organisms.

## 2. Key Features

- Underwater image enhancement using U-Net.
- Marine organism detection using a YOLO-based model.
- Fish species classification using a trained EfficientNet model.
- Confidence-based detection evaluation.
- Rule-based feedback controller for handling uncertain detection results.
- Alternative image-processing and detection attempts, subject to configured retry limits.
- Bounding-box visualization and species prediction display.
- Grad-CAM-based model explainability, where implemented.

## 3. Project Files

| File | Description |
|---|---|
| `index.py` | Updated implementation incorporating the feedback-controller workflow. |
| `main.py` | Previous implementation, retained for comparison. |

**Recommended execution file:** `index.py`

## 4. Deep Learning Models

### 4.1 U-Net — Image Enhancement

U-Net is used to process underwater images and improve the visibility of relevant visual features. The enhanced image can provide an alternative input for subsequent object detection.

**Purpose:** Improve image quality before detection.

### 4.2 YOLO — Marine Organism Detection

The YOLO-based object detection model identifies marine organisms and predicts their locations using bounding boxes. It produces detection labels and confidence scores used by the feedback controller.

**Purpose:** Detect and localize fish and other supported marine organisms.

### 4.3 EfficientNet — Fish Species Classification

The trained EfficientNet-based classification model analyses detected fish regions and predicts their species based on learned visual features.

**Purpose:** Classify detected fish into the species represented in the trained model.

### 4.4 Rule-Based Feedback Controller

The feedback controller connects detection evaluation with subsequent processing. It checks the quality of the detection results and determines whether they can proceed to classification or require another processing attempt.

For candidate \(i\), the conceptual evaluation score is:

\[
S_i=w_C C_i+w_A A_i+w_H H_i
\]

where:

- \(S_i\): Candidate evaluation score.
- \(C_i\): Mean detection confidence.
- \(A_i\): Agreement between detections at different processing scales.
- \(H_i\): Fraction of high-confidence detections.
- \(w_C,w_A,w_H\): Weights assigned to the evaluation criteria.

The actual formula and weights must match the implementation in `index.py`. The controller uses predefined rules and a configured retry limit rather than a conventional PID controller.

### 4.5 Grad-CAM — Model Explainability

Grad-CAM can generate visual explanations that highlight image regions influencing a model's classification prediction.

**Purpose:** Help interpret classification decisions and visualize relevant image features.


## 5. System Architecture

```text
          Input Underwater Image
                    |
                    v
          U-Net Image Enhancement
                    |
                    v
          YOLO Object Detection
                    |
                    v
          Detection Quality Evaluation
                    |
                    v
          Rule-Based Feedback Controller
                    |
           +--------+--------+
           |                 |
      Weak Results      Acceptable Results
           |                 |
           v                 v
     Alternative       Fish Region Extraction
      Processing              |
           |                  v
           v          EfficientNet Classification
       Re-detection            |
           |                   v
           +----------> Species Predictions
                               |
                               v
                      Final Visualization
```

The diagram represents the intended workflow. The actual processing order and retry logic should reflect the code in `index.py`.

## 6. Technologies Used

- Python
- PyTorch and/or TensorFlow/Keras, depending on model implementation
- U-Net
- YOLO-based object detection
- EfficientNet-based classification
- OpenCV
- NumPy
- Pandas
- Grad-CAM, where implemented

## 7. Dataset Usage

The project is designed to use multiple datasets for training and evaluating the relevant models.

Dataset usage should be documented according to the actual training pipeline:

- **Image enhancement:** Images used to train or evaluate U-Net.
- **Object detection:** Annotated images used to train or evaluate the YOLO-based detector.
- **Species classification:** Labelled fish images used to train or evaluate EfficientNet.


## 8. Installation and Execution

### Prerequisites

- Python environment compatible with the installed dependencies.
- Required trained model weights.
- Project source files and configuration.
- Input images for testing.

### Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

### Run the updated project

```bash
python index.py
```

### Run the previous version

```bash
python main.py
```

The updated version should be used to demonstrate the feedback-controller workflow.

## 9. Expected Results

Depending on the implemented features, the system can produce:

- Enhanced underwater images.
- Bounding boxes around detected organisms.
- Detection confidence scores.
- Predicted fish species.
- Feedback-based detection results.
- Grad-CAM visualizations, if enabled.

Prediction quality depends on the training datasets, model weights, image quality, and similarity between species.

## 10. Limitations

- Low visibility and poor lighting can affect detection.
- Overlapping organisms may produce missed or duplicate detections.
- Similar-looking fish species may be misclassified.
- Image enhancement does not guarantee better detection in every case.
- The feedback controller can request additional processing but cannot guarantee correct predictions.

## 11. Future Scope

- Expand the dataset to include additional fish species and environmental conditions.
- Improve detection and classification performance through systematic evaluation.
- Extend tracking and population-monitoring capabilities.
- Integrate growth-stage prediction and water-quality assessment using suitable trained models.
- Evaluate the feedback controller using precision, recall, F1-score, and retry statistics.

## 12. Conclusion

The Closed-Loop Marine Monitoring System integrates U-Net-based image enhancement, YOLO-based object detection, EfficientNet-based fish species classification, and a rule-based feedback controller.

The updated `index.py` implementation is intended to provide the main feedback-based workflow, while `main.py` preserves the previous implementation for comparison. Together, these components form a modular deep learning pipeline for automated marine organism detection and fish species monitoring.
