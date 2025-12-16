# Report Completion Guide

Reviewing your `Report_template.tex` and your project files, you have a strong foundation. The Abstract and Introduction are well-written and align with your project ("MedAnomaly Suite").

Here is a step-by-step plan to complete the report by filling in the missing sections with content from your actual implementation.

## 1. Data Section
**Goal**: Describe the dataset used effectively.
- **Source**: Kaggle Brain Tumor MRI Dataset (Masoud Nickparvar).
- **Details**:
    - **Classes**: glioma, meningioma, pituitary, no_tumor.
    - **Quantity**: Mention the total number of images (check your `count_data.py` output if known, or estimating from standard dataset size ~7000 images).
    - **Preprocessing**: Mention resizing to 256x256, normalization (0-1 or -1 to 1), and train/val/test splits.

## 2. Methods Section
**Goal**: Technical deep-dive into your architecture. You have three main components to describe:

### A. VQ-VAE / VQ-GAN (Anomaly Detection)
*References: `src/model/vqgan.py`, `src/model/quantizer.py`*
- Explain the **Encoder**: Compresses image to latent space.
- Explain the **Vector Quantizer**: Discretizes the latent space (Codebook).
- Explain the **Decoder**: Reconstructs the image from indices.
- **Loss Functions**: Mention Reconstruction Loss (L1/L2) + Perceptual Loss (LPIPS from `src/model/perceptual_loss.py`) + Adversarial Loss (`discriminator.py`).
- **Anomaly Detection Logic**: Explain how you use Reconstruction Error (SSIM/MSE) to find the tumor.

### B. Transformer (Counterfactual Generation)
*References: `src/model/transformer.py`*
- Explain it's a GPT-style transformer (MinGPT based likely).
- **Purpose**: Autoregressively predicts "healthy" codebook indices given context, enabling "Dream Mode" (generating what the brain *should* look like without the tumor).

### C. Classifier (Diagnosis)
*References: `src/model/classifier.py`*
- Architecture: ResNet18 (Transfer learning or trained from scratch).
- Head: 4-class Softmax layer.

## 3. Experiments & Results
**Goal**: Show that it works. Replace the template placeholders.

### Figures to Include
Use the images you already have in your root directory!
- **`inference_result.png`**: Perfect for the "Anomaly Detection" qualitative results. Shows Input vs Reconstruction vs Heatmap.
- **`dream_patient.png`**: Perfect for the "Counterfactual/Dream Mode" section. Shows Tumor vs Generated Healthy.
- **`final_report_card.png`**: Good for showing the end-to-end "Product" view (Dashboard).
- **`learning.png`**: Generate a loss curve plot from your training logs if you have them (e.g. from `outputs/` or TensorBoard).

### Metrics Table
- **Classifier Accuracy**: 99.08% (as noted in your README).
- **Anomaly Detection**: If you calculated AUC-ROC or SSIM scores, list them.
- **Inference Speed**: Mention it runs on M-Series chips (Apple Silicon) locally.

## 4. Latex Tips
- **Images**: Use the `\includegraphics` commands currently in the template but point them to your actual files.
    - *Example*: `\includegraphics[width=1.0\linewidth]{inference_result.png}`
- **Citations**: Ensure you cite the VQ-VAE paper (Van den Oord et al.) and VQGAN (Esser et al.) in the related works or methods.

## Example "Methods" Snippet (Draft)
```latex
\subsection{Vector Quantized GAN (VQ-GAN)}
We employ a VQ-GAN architecture to learn a discrete latent representation of healthy brain anatomy. The model consists of an encoder $E$, a decoder $G$, and a codebook $Z \in \mathbb{R}^{K \times D}$. The encoder maps input $x$ to spatial latents...

\subsection{Latent Transformer}
To model the global distribution of healthy latent codes, we train a transformer $T$ autoregressively on the indices produced by the quantization step...
```
