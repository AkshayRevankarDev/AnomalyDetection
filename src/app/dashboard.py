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
    page_title="MedAnomaly | Neuro-Oncology Suite",
    page_icon="⚕️",
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
        st.warning("System Alert: Classifier weights not found. Using random initialization.")
    
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
st.sidebar.title("MedAnomaly Platform")
st.sidebar.markdown("**System Status:** Online")
st.sidebar.markdown("---")

st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload DICOM/MRI Sequence", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Anomaly Detection Sensitivity", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("Research Tools")
st.sidebar.markdown("**Synthetic Data Generation**")
if st.sidebar.button("Generate Synthetic Subject"):
    with st.spinner("Synthesizing neural features..."):
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
            
            st.sidebar.image(fake_img_np, caption="Synthetic Output [Generated]", use_container_width=True)
            st.sidebar.success("Generation Complete")

# Main Layout
st.title("Neuro-Oncology Diagnostic Suite")
st.markdown("### Automated Anomaly Detection & Classification System")
st.markdown("---")

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert('L')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quantitative Anomaly Analysis")
        # Run Anomaly Detection
        # Save temp file for detector (it expects path)
        temp_path = "temp_upload.png"
        image.save(temp_path)
        
        original, recon, raw_diff, ssim_map, smoothed_map = detector.detect(temp_path)
        
        # Display Side-by-Side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original, cmap='gray')
        ax[0].set_title("Input Sequence (T1-Weighted)")
        ax[0].axis('off')
        
        ax[1].imshow(smoothed_map, cmap='hot')
        ax[1].set_title("Anomaly Localization Map")
        ax[1].axis('off')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Differential Diagnosis")
        
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
        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
        predicted_class = classes[predicted.item()]
        confidence = probs[0][predicted.item()].item()
        
        # Get Recommendations
        # Map class names to keys expected by ClinicalLogic if needed, or update ClinicalLogic
        # Assuming ClinicalLogic expects lowercase
        plan_key = predicted_class.lower().replace(" ", "")
        # The previous mapping was: 'glioma', 'meningioma', 'notumor', 'pituitary'
        # Let's map back to that for the logic call
        logic_key_map = {
            'Glioma': 'glioma',
            'Meningioma': 'meningioma',
            'No Tumor': 'notumor',
            'Pituitary Tumor': 'pituitary'
        }
        
        plan = ClinicalLogic.get_treatment_plan(logic_key_map[predicted_class])
        
        # Display Diagnosis
        # Professional coloring
        st.markdown(f"**Predicted Pathology:**")
        st.markdown(f"<h3 style='color:#2c3e50;'>{predicted_class}</h3>", unsafe_allow_html=True)
        st.write(f"**Model Confidence:** {confidence:.4f}")
        
        # Grad-CAM Visualization
        st.markdown("#### Model Interpretability (Grad-CAM)")
        gradcam = GradCAM(classifier.model, classifier.model.layer4)
        heatmap = gradcam(input_tensor, class_idx=predicted.item())
        
        img_np = input_tensor.squeeze().cpu().numpy()
        overlay = GradCAM.overlay_heatmap(img_np, heatmap, alpha=0.4)
        
        st.image(overlay, caption="Class Activation Map (Input Space)", use_container_width=True)
        
        st.markdown("---")
        
        # Display Plan
        st.markdown("#### Clinical Decision Support")
        st.info(f"**Risk Stratification:** {plan['Risk Level']}")
        st.write(f"**Action Protocol:** {plan['Immediate Actions']}")
        st.write(f"**Longitudinal Care:** {plan['Follow-up']}")
        
        # Probability Bar Chart
        st.markdown("---")
        st.markdown("**Probabilistic Distribution:**")
        prob_dict = {classes[i]: probs[0][i].item() for i in range(4)}
        st.bar_chart(prob_dict)

else:
    st.info("System Ready. Please initialize analysis by uploading a patient scan.")
