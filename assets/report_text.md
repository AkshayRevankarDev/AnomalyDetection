# The AI Radiologist: A Unified Generative & Diagnostic Suite for Neuro-Oncology

**Team:** MedAnomaly Group (Group #22)
**Hardware:** Apple Silicon M4 Pro (MPS Optimized)

## Abstract
In the critical field of Neuro-Oncology, the "Black Box" nature of Artificial Intelligence and rigorous patient privacy laws often impede the adoption of high-performance diagnostic tools. To address this, we present "The AI Radiologist," a unified suite combining unsupervised anomaly detection with explainable supervised classification. Our approach leverages a hybrid VQ-GAN (Vector Quantized Generative Adversarial Network) for high-fidelity reconstruction and anomaly localization, a ResNet18 classifier with Grad-CAM for transparent diagnosis, and a Latent Transformer for privacy-preserving synthetic data generation. Experiments on a 4-class Brain MRI dataset demonstrate the system's ability to robustly detect anomalies, classify tumor types with high accuracy, and "dream" realistic synthetic patient data, proving the viability of such a comprehensive system on consumer hardware like the Apple Silicon M4 Pro.

## 1. Introduction
**Problem:** The integration of AI into clinical radiology faces two major hurdles: trust and privacy. Clinicians are hesitant to rely on "Black Box" models that offer no explanation for their diagnoses, making misdiagnosis a fatal possibility. Furthermore, strict patient privacy regulations (HIPAA, GDPR) severely restrict the sharing of medical imaging data, stalling research progress.

**Relevance:** Brain tumors (Glioma, Meningioma, Pituitary) require precise and timely intervention. A system that can accurately diagnose, explain its reasoning, and generate shareable synthetic data could significantly accelerate research and improve patient outcomes without compromising privacy.

**Merit:** Our work bridges the gap between unsupervised and supervised learning. By employing VQ-GAN, we achieve superior reconstruction quality compared to standard VQ-VAEs, allowing for the detection of subtle anomalies that a supervised classifier might miss if not explicitly trained on them. We further enhance trust by overlaying Grad-CAM attention maps, showing exactly *where* the model is looking.

## 2. Related Works
Our system builds upon and extends several foundational architectures:
- **Van den Oord et al. (2017) (VQ-VAE):** Introduced discrete latent representations. We improve upon this by adding adversarial loss for sharper reconstructions.
- **Esser et al. (2021) (VQ-GAN/Taming Transformers):** The core architecture for our visual pipeline. By using a discriminator and perceptual loss, we overcome the blurriness inherent in standard VAEs.
- **Selvaraju et al. (2017) (Grad-CAM):** We integrate this to provide visual explanations for our ResNet18 classifier's decisions.
- **Ronneberger et al. (2015) (U-Net):** The standard for segmentation. We demonstrate that generative approaches (VQ-GAN) offer valid alternatives for anomaly detection without pixel-level labels.
- **Vaswani et al. (2017) (Transformers):** The backbone of our "Dream Mode," enabling the system to understand the "language" of brain MRIs and generate new samples.

## 3. Data
**Source:** We utilized Masoud Nickparvar’s Brain Tumor MRI Dataset from Kaggle.
**Details:** The dataset consists of approximately 7,000 MRI scans divided into four classes: Glioma, Meningioma, Pituitary, and No Tumor.

**Preprocessing:** A critical challenge was the choice of image transformation.
- *CenterCrop:* Initially, standard `CenterCrop` cut off outer skull regions, removing vital context.
- *Full Resize:* We switched to `Resize((128, 128))`, effectively "zooming out" to preserve the full skull shape. This ensured the model learned global geometry rather than just local texture.
- **3.1 Visual Data:** Refer to **Figure 1** (generated separately) which illustrates the "CenterCrop" artifact versus our "Full Resize" correction.

**3.2 Tabular Results:**
Below are the classification metrics achieved by our fine-tuned ResNet18 model on the test set.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Glioma | 0.97 | 0.96 | 0.96 | 300 |
| Meningioma | 0.96 | 0.98 | 0.97 | 306 |
| Pituitary | 0.99 | 0.98 | 0.98 | 300 |
| No Tumor | 0.98 | 0.98 | 0.98 | 405 |
| **Accuracy** | | | **0.98** | **1311** |

## 4. Methods
**End-to-End Pipeline:**
1.  **Input:** A Brain MRI scan (Grayscale, resized to 128x128).
2.  **VQ-GAN Reconstruction:** The image is encoded into discrete codebook indices (16x16 grid) and reconstructed. The difference between input and reconstruction yields the "Anomaly Heatmap."
3.  **ResNet Classification:** The image is passed to a ResNet18 classifier to predict the tumor type.
4.  **Grad-CAM Overlay:** Gradients from the final convolutional layer are visualized to highlight the tumor region.
5.  **Latent Transformer Synthesis:** In "Dream Mode," a Transformer predicts a sequence of codebook indices, which the VQ-GAN decoder converts into a synthetic brain image.

**Justification:**
- **VQ-GAN over VAE:** Medical anomalies are often subtle. VQ-GAN's perceptual loss (LPIPS) ensures that reconstructions match human perceptual quality, preventing "blur" from masking small tumors.
- **Transformers over GANs:** Predicting discrete tokens allows for better modeling of the global structure of the brain compared to the potentially unstable adversarial training of pure GANs.

## 5. Experiments and Results
**Training:** The VQ-GAN was trained for 100 epochs on an Apple Silicon M4 Pro (MPS backend).
**Evaluation:**
- **Reconstruction Fidelity:** The model learned to reconstruct complex brain structures, including ventricles and gyri, with high sharpness.
- **Diagnostic Accuracy:** The system achieved near-perfect classification, with 98% accuracy across all classes. Meningioma detection was particularly robust.

**Figure 2:** Please refer to the "Training vs Validation Loss" plot generated below, which demonstrates the model's convergence stability.

## 6. Limitations
- **2D Slices vs 3D Volumetric:** Our analysis treats each slice independently, ignoring the rich 3D context of a full MRI volume. Small tumors visible in only one slice might be missed without adjacent context.
- **Data Quality Dependence:** The generative model is sensitive to training data quality. Noise or artifacts in the original dataset can form part of the "learned" distribution, leading to imperfect reconstructions.

## 7. Conclusion
**Summary:** We have successfully built a "God Tier" local AI dashboard that democratizes advanced neuro-oncology tools. By combining generative and discriminative AI, we offer a solution that is both accurate and transparent.
**Future Work:** We plan to extend this architecture to support native 3D DICOM formats and implement Federated Learning to allow privacy-preserving training across multiple institutions.
