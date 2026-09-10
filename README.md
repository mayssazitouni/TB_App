🧠 Tuberculosis Detection using CNN and Grad-CAM
🎯 Objective

This project implements an end-to-end deep learning solution to detect Tuberculosis (TB) from chest X-ray images, providing real-time AI-assisted medical diagnosis and visual explainability.

Key components:

CNN Classification Model trained to analyze chest X-ray images and classify them as Normal or Tuberculosis

Grad-CAM Heatmap Visualization to highlight pathological lung regions driving the model's decision

FastAPI Backend Services to handle inference and image processing asynchronously

Interactive Streamlit Web Dashboard featuring dark mode for intuitive clinical analysis

---

## 📁 Dataset

- **Source**: [TB Chest Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-chest-xray-dataset)
- **Total Images**: 4200
  - **Normal**: 3500
  - **Tuberculosis**: 700
- **Preprocessing**:
  - Resized to **150x150 pixels**
  - Pixel normalization to `[0, 1]`
  - Data augmentation applied to training set (rotation, zoom, shift, flip)

  ---

## ⚖️ Handling Class Imbalance

The dataset is imbalanced (Normal: 3500, Tuberculosis: 700). To address this:

* **Class Weights**: Computed using `sklearn.utils.class_weight` and passed to the model during training to give more importance to the minority class.
* **Data Augmentation**: Applied to the training set to increase diversity and reduce overfitting on the dominant class.

These strategies helped the model achieve high recall on tuberculosis cases, which is critical in medical diagnostics.


---

## 🧱 Model Architecture

* **Type**: Custom CNN built with TensorFlow/Keras
* **Input Shape**: `(150, 150, 3)`
* **Layers**:
  i. `Conv2D(32)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D`
  ii. `Conv2D(64)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D`
  iii. `Conv2D(128)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D`
  iv. `Flatten`
  v. `Dense(128, relu)` $\rightarrow$ `Dropout(0.5)`
  vi. `Dense(1, sigmoid)`
* **Optimizer**: `Adam` with learning rate `0.0001`
* **Loss Function**: `binary_crossentropy`
* **Metrics**: Accuracy, Precision, Recall, AUC

---

## 📊 Final Results

After training for 10 epochs with callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`), the model achieved:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 98.3% |
| **Recall (Tuberculosis)** | 94.3% |
| **Precision (Tuberculosis)** | 95.7% |
| **F1-Score (Tuberculosis)** | 0.95 |
| **Validation Loss** | 0.0992 |

> **Note**: On the test set of 840 images, the model correctly identified **132 out of 140** Tuberculosis cases (8 false negatives) and **694 out of 700** Normal cases (6 false positives).

---

## 🔍 Explainable AI (Grad-CAM)

To interpret the model's decisions, Grad-CAM heatmaps are generated for validation images. These visualizations highlight the lung regions that influenced the prediction, helping build trust in the model's output.

* **Implemented using TensorFlow and OpenCV**
* **Displays both true and predicted labels with confidence scores**
* **Heatmaps are overlaid on original X-ray images**

---

## ⚙️ Technologies Used

* **Python**
* **TensorFlow & Keras**
* **NumPy & Matplotlib**
* **OpenCV (Grad-CAM)**
* **Scikit-learn (metrics)**
* **FastAPI & Uvicorn (Backend API)**
* **Streamlit (Frontend UI)**

---

## 📁 Project Structure

```text
TB_App/
├── api.py                    # FastAPI backend server
├── Front.py                  # Streamlit frontend user interface
├── tb_cnn_model_best.keras   # Trained CNN model weights
├── start.bat                 # Automated startup script for Windows
├── requirements.txt          # Python project dependencies
├── .gitignore                # Files and folders ignored by Git
└── .streamlit/
    └── config.toml           # Streamlit dark theme configuration

    ---

## 👩‍💻 Author

**Mayssa Zitouni**  
Computer Engineering Student  
Focused on AI for healthcare