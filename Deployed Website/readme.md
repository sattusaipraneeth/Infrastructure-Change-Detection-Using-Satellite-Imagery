# 🛰️ Change Detection using Siamese U-Net  

This repository provides a **Streamlit-based web application** for detecting changes between two images using a **Siamese U-Net model**. The model is designed for **infrastructure monitoring, disaster assessment, and environmental change detection** by analyzing satellite or aerial images.  

The application takes two images (**pre-change** and **post-change**) as input and generates a **segmentation map** highlighting the detected changes.  

## 🚀 Features  
✔️ **Siamese U-Net Architecture** for precise change detection  
✔️ **Streamlit Web Interface** for an interactive user experience  
✔️ **Automatic Preprocessing** (Resizing & Normalization)  
✔️ **Real-time Segmentation Mask Generation**  
✔️ **CPU-Friendly Inference** with PyTorch  

## 📜 Model Overview  
The **Siamese U-Net** consists of two parallel U-Net branches with **shared weights**, encoding both input images separately before fusing features for decoding. This approach enhances **feature comparison** and ensures **accurate change detection**.  

## 📂 File Structure  
```plaintext
📦 siamese-unet-change-detection  
 ┣ 📜 app.py                 # Streamlit Web App  
 ┣ 📜 siamese_unet.pth       # Pretrained Model - Use any model from download links in Repository  
 ┣ 📜 README.md              # Documentation  
 ┗ 📜 requirements.txt       # Dependencies  
