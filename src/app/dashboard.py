import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.engine.anomaly_detector import AnomalyDetector
from src.model.classifier import TumorClassifier
from src.analytics.recommendations import ClinicalLogic
from src.analytics.gradcam import GradCAM
from src.model.transformer import LatentTransformer
from src.model.vqgan import VQGAN
import cv2

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
    # 1. Anomaly Detector (VQ-GAN)
    detector = AnomalyDetector(config, model_path="checkpoints/vqgan_best.pth")
    
    # 2. Tumor Classifier (ResNet18)
    classifier = TumorClassifier(num_classes=4)
    # Check if weights exist
    if os.path.exists("checkpoints/diagnosis_resnet.pth"):
        classifier.load_state_dict(torch.load("checkpoints/diagnosis_resnet.pth", map_location=config['training']['device']))
    else:
        st.warning("Classifier weights not found. Using random weights.")
    
    classifier.to(config['training']['device'])
    classifier.eval()
    
    # 3. Latent Transformer
    transformer = LatentTransformer(config['transformer'])
    if os.path.exists("checkpoints/transformer_best.pth"):
        transformer.load_state_dict(torch.load("checkpoints/transformer_best.pth", map_location=config['training']['device']))
    
    transformer.to(config['training']['device'])
    transformer.eval()
    
    return detector, classifier, transformer

detector, classifier, transformer = load_models()

# Sidebar
st.sidebar.title("🩻 AI Radiologist")
st.sidebar.info("MedAnomaly Suite V3.0")
uploaded_file = st.sidebar.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5)

st.sidebar.divider()
st.sidebar.subheader("✨ Generative AI")
if st.sidebar.button("Dream Mode (Generate Patient)"):
    with st.spinner("Dreaming..."):
        # Generate
        device = config['training']['device']
        idx = torch.randint(0, config['transformer']['vocab_size'], (1, 1)).to(device)
        
        with torch.no_grad():
            generated_indices = transformer.generate(idx, max_new_tokens=255, temperature=1.0)
            
            # Decode with VQGAN (Accessing internal model from detector)
            vqgan = detector.model
            z_q = vqgan.quantizer._embedding(generated_indices).view(1, 16, 16, 128)
            z_q = z_q.permute(0, 3, 1, 2)
            
            fake_img = vqgan.decoder(z_q)
            fake_img_np = fake_img.squeeze().cpu().numpy()
            
            st.sidebar.image(fake_img_np, caption="Synthetic Patient", use_container_width=True)
            st.sidebar.success("Generated!")

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
        
        # Grad-CAM Visualization
        st.write("### 🔍 Explainable AI (Grad-CAM)")
        gradcam = GradCAM(classifier.model, classifier.model.layer4)
        heatmap = gradcam(input_tensor, class_idx=predicted.item())
        
        # Overlay heatmap on original image
        # Resize image to match heatmap if needed, but GradCAM output is already resized to input size (128x128)
        # We need to use the original image (which might be different size) or the transformed tensor
        # Let's use the transformed tensor for consistency
        img_np = input_tensor.squeeze().cpu().numpy()
        overlay = GradCAM.overlay_heatmap(img_np, heatmap, alpha=0.4)
        
        st.image(overlay, caption="Model Attention Map", use_container_width=True)
        
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
