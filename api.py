import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

app = FastAPI(title="TB Chest Radiography API avec Grad-CAM")

# Configuration des constantes
IMG_SIZE = (150, 150)
MODEL_PATH = "tb_cnn_model_best.keras"

# Chargement du modèle
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# ==============================================================================
#                      FONCTIONS GRAD-CAM DU NOTEBOOK
# ==============================================================================

def find_last_conv_layer(model):
    """
    Trouve le nom de la dernière couche Conv2D du modèle.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Aucune couche Conv2D trouvée dans le modèle.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, y_pred_label):
    """
    Génère la heatmap Grad-CAM par reconstruction du graphe.
    """
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x

    grad_model = tf.keras.Model(inputs, [conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if y_pred_label == 1:
            class_channel = predictions[:, 0]
        else:
            class_channel = 1 - predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_heatmap(heatmap, img_bgr, alpha=0.4):
    """
    Superpose la heatmap Grad-CAM sur l'image d'origine.
    """
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)

# ==============================================================================
#                            LOGIQUE PRÉDICTION / API
# ==============================================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    contents = await file.read()
    
    # 1. Chargement et prétraitement PIL
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    pil_resized = pil_img.resize(IMG_SIZE)
    
    # Format Tenseur pour le modèle (Normalisé 1./255)
    img_tensor = np.expand_dims(np.array(pil_resized, dtype=np.float32) / 255.0, axis=0)
    
    # 2. Prédiction
    pred_score = float(model.predict(img_tensor, verbose=0)[0][0])
    y_pred_label = 1 if pred_score >= 0.5 else 0
    
    label = "Tuberculosis" if y_pred_label == 1 else "Normal"
    confidence = pred_score if y_pred_label == 1 else (1.0 - pred_score)

    # 3. Génération Grad-CAM avec OpenCV
    try:
        last_conv = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(img_tensor, model, last_conv, y_pred_label)
        
        # Image d'origine BGR aux dimensions de l'image originale
        img_np_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        
        # Superposition avec ta fonction
        overlaid_bgr = overlay_heatmap(heatmap, img_bgr)
        
        # Encodage en JPEG Base64 pour transmission HTTP vers Streamlit
        _, buffer = cv2.imencode('.jpg', overlaid_bgr)
        gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Erreur Grad-CAM : {e}")
        gradcam_base64 = None

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(pred_score, 4),
        "gradcam": gradcam_base64
    }