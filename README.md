GAN-Based Synthetic Iris Image Generation (Visible & NIR)
📌 Overview
This repository implements a Conditional DCGAN (cDCGAN) pipeline for generating synthetic iris images and evaluating their impact on classification performance.
The project explores:
•	Synthetic data generation for Visible (RGB) iris images
•	Synthetic data generation for NIR (grayscale) iris images
•	Challenges in low-texture NIR generation
•	Improvement using CLAHE preprocessing
•	Evaluation using classifiers and image quality metrics
 
📂 Project Structure
src/
│
├── GAN Models
│   └── cDCGAN_model_v3.py
│
├── GAN Training
│   ├── train_cDCGAN_visible_v4.py
│   ├── train_cDCGAN_NIR_uncropped.py
│
├── Synthetic Data Generation
│   ├── generate_synthetic_visible_data.py
│   ├── generate_synthetic_nir_data.py
│   ├── generate_NIR_samples.py
│
├── Preprocessing
│   ├── preprocess_NIR_CLAHE.py
│   ├── preprocess_iris_enhanced.py
│   ├── crop_image.py
│   ├── manual_iris_crop.py
│   ├── manual_circular_crop_simple.py
│   ├── manual_blackout_irregular.py
│
├── Dataset Utilities
│   ├── split_visible_dataset.py
│   ├── split_dataset_by_class.py
│   ├── analyze_dataset_stats.py
│
├── Classifier Training & Evaluation
│   ├── train_and_evaluate_visible_classifiers.py
│   ├── train_vit_only_visible_classifier.py
│   ├── evaluate_classifier_visible_autosave.py
│   ├── generate_classifier_predictions_visible.py
│   ├── analyze_classifier_results_visible.py
│
├── Evaluation Metrics
│   ├── evaluate_fid_ssim_tsne_visible.py
│
├── Debug / Utility
│   └── verify_visible_dataloader.py
 
⚙️ Installation
1. Create virtual environment
python3 -m venv venv_iris
source venv_iris/bin/activate
2. Install dependencies
pip install -r requirements.txt
 
🚀 Workflow
🔹 Step 1: Preprocess NIR Images (CLAHE)
python preprocess_NIR_CLAHE.py
 
🔹 Step 2: Train GANs
Visible (RGB)
python train_cDCGAN_visible_v4.py
NIR (CLAHE, uncropped)
python train_cDCGAN_NIR_uncropped.py
 
🔹 Step 3: Generate Synthetic Data
Visible
python generate_synthetic_visible_data.py
NIR
python generate_synthetic_nir_data.py
 
🔹 Step 4: Train Classifiers
python train_and_evaluate_visible_classifiers.py
Optional (ViT only):
python train_vit_only_visible_classifier.py
 
🔹 Step 5: Evaluate Results
python evaluate_classifier_visible_autosave.py
python evaluate_fid_ssim_tsne_visible.py
 
🧠 Key Observations
•	Visible GAN produces high-quality, diverse iris images
•	NIR GAN (baseline) struggles due to low texture
•	CLAHE preprocessing improves NIR generation stability
•	Synthetic data improves classification — but must be realistic
 
⚠️ Challenges
•	Mode collapse in grayscale GAN training
•	Low contrast in NIR images
•	High-resolution training on MPS (Mac) requires small batch sizes
 
📚 Dataset
•	Warsaw-BioBase-Disease-Iris Dataset
(Visible + NIR bands)
 
👤 Author
Ananya Zabin
 
📜 License
For academic and research use only.
<img width="468" height="642" alt="image" src="https://github.com/user-attachments/assets/2fb4d4f6-f44e-4628-b348-76b9289105a4" />
