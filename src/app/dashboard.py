import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from src.engine.anomaly_detector import AnomalyDetector
from src.model.classifier import TumorClassifier
from src.analytics.recommendations import ClinicalLogic

# Page Config
st.set_page_config(
    page_title="AI Radiologist - MedAnomaly Suite V3.0",
    page_icon="🧠",
    layout="wide"
)

# Load Config
def load_config():
    with open("configs/config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Load Models (Cached)
@st.cache_resource
def load_models():
    # 1. Anomaly Detector (VQ-VAE)
    detector = AnomalyDetector(config, model_path="checkpoints/vqvae_best.pth")
    
    # 2. Tumor Classifier (ResNet18)
    classifier = TumorClassifier(num_classes=4)
    # Check if weights exist
    if os.path.exists("checkpoints/diagnosis_resnet.pth"):
        classifier.load_state_dict(torch.load("checkpoints/diagnosis_resnet.pth", map_location=config['training']['device']))
    else:
        st.warning("Classifier weights not found. Using random weights.")
    
    classifier.to(config['training']['device'])
    classifier.eval()
    
    return detector, classifier

detector, classifier = load_models()

# Sidebar
st.sidebar.title("🩻 AI Radiologist")
st.sidebar.info("MedAnomaly Suite V3.0")
uploaded_file = st.sidebar.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5)

# Main Layout
st.title("🧠 Neuro-Oncology Dashboard")

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert('L')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scan Analysis")
        # Run Anomaly Detection
        # Save temp file for detector (it expects path)
        temp_path = "temp_upload.png"
        image.save(temp_path)
        
        original, recon, raw_diff, ssim_map, smoothed_map = detector.detect(temp_path)
        
        # Display Side-by-Side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original, cmap='gray')
        ax[0].set_title("Original Scan")
        ax[0].axis('off')
        
        ax[1].imshow(smoothed_map, cmap='hot')
        ax[1].set_title("Anomaly Heatmap")
        ax[1].axis('off')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Diagnosis & Plan")
        
        # Run Classification
        # Transform image for classifier
        from torchvision import transforms
        transform = TumorClassifier.get_transforms()
        input_tensor = transform(image).unsqueeze(0).to(config['training']['device'])
        
        with torch.no_grad():
            outputs = classifier(input_tensor)
            _, predicted = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        # Classes (Hardcoded based on dataset knowledge or loaded)
        # Usually: ['glioma', 'meningioma', 'notumor', 'pituitary']
        # We need to ensure mapping is correct. 
        # For now, let's assume alphabetical order which `ImageFolder` uses.
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        predicted_class = classes[predicted.item()]
        confidence = probs[0][predicted.item()].item()
        
        # Get Recommendations
        plan = ClinicalLogic.get_treatment_plan(predicted_class)
        
        # Display Diagnosis
        color = "red" if plan['Risk Level'] == "High" else "orange" if plan['Risk Level'] == "Medium" else "green"
        st.markdown(f"<h2 style='color:{color};'>{plan['Diagnosis']}</h2>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.2%}")
        
        st.divider()
        
        # Display Plan
        st.markdown("### 📋 Clinical Recommendations")
        st.write(f"**Risk Level:** {plan['Risk Level']}")
        st.write(f"**Immediate Action:** {plan['Immediate Actions']}")
        st.write(f"**Follow-up:** {plan['Follow-up']}")
        
        # Probability Bar Chart
        st.divider()
        st.write("### Class Probabilities")
        prob_dict = {classes[i]: probs[0][i].item() for i in range(4)}
        st.bar_chart(prob_dict)

else:
    st.info("Please upload a Brain MRI scan to begin analysis.")
