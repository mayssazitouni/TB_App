import streamlit as st
import requests
import base64
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="Détection de Tuberculose (IA)",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Analyse de Radiographie Thoracique")
st.write("Téléchargez une image X-Ray pour tester la présence de Tuberculose avec diagnostic visuel Grad-CAM.")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Choisissez une image de radio...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image importée", use_container_width=True)

    if st.button("Lancer le Diagnostic", type="primary"):
        with st.spinner("Analyse de la radiographie par l'IA..."):
            try:
                bytes_data = uploaded_file.getvalue()
                files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}

                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    gradcam_b64 = result.get("gradcam")

                    st.divider()

                    if prediction == "Tuberculosis":
                        st.error(f"🚨 **Résultat : Tuberculose détectée**")
                        st.warning(f"Indice de confiance : **{confidence}%**")
                    else:
                        st.success(f"✅ **Résultat : Normal (Sain)**")
                        st.info(f"Indice de confiance : **{confidence}%**")

                    if gradcam_b64:
                        st.divider()
                        st.markdown("### 🔥 Visualisation d'importance Grad-CAM")
                        
                        image_data = base64.b64decode(gradcam_b64)
                        gradcam_img = Image.open(io.BytesIO(image_data))
                        
                        st.image(gradcam_img, caption="Zones d'activité du modèle (Grad-CAM)", use_container_width=True)
                    else:
                        st.info("Visualisation Grad-CAM non disponible.")

                else:
                    st.error("Erreur provenant du serveur API.")

            except Exception as e:
                st.error(f"Impossible de contacter l'API : {e}")