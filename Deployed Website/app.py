import streamlit as st
import torch
import numpy as np
import cv2
import requests
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        p = self.pool(x)
        return x, p  # Return both the feature map and pooled result

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)  # Account for concatenation
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class SiameseUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SiameseUNet, self).__init__()

        # Encoder blocks for image 1
        self.encoder1a = EncoderBlock(in_channels, 64)
        self.encoder2a = EncoderBlock(64, 128)
        self.encoder3a = EncoderBlock(128, 256)
        self.encoder4a = EncoderBlock(256, 512)

        # Encoder blocks for image 2
        self.encoder1b = EncoderBlock(in_channels, 64)
        self.encoder2b = EncoderBlock(64, 128)
        self.encoder3b = EncoderBlock(128, 256)
        self.encoder4b = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(1024, 1024)  # 512 + 512 from both encoders

        self.decoder4 = DecoderBlock(1024, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        # Final convolution to output the segmentation map
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Encoder part for image 1
        e1a, p1a = self.encoder1a(x1)
        e2a, p2a = self.encoder2a(p1a)
        e3a, p3a = self.encoder3a(p2a)
        e4a, p4a = self.encoder4a(p3a)

        # Encoder part for image 2
        e1b, p1b = self.encoder1b(x2)
        e2b, p2b = self.encoder2b(p1b)
        e3b, p3b = self.encoder3b(p2b)
        e4b, p4b = self.encoder4b(p3b)

        # Concatenate the encoder outputs from both images (for each layer)
        concat_e4 = torch.cat([e4a, e4b], dim=1)  # Concatenate along the channel dimension
        concat_e3 = torch.cat([e3a, e3b], dim=1)
        concat_e2 = torch.cat([e2a, e2b], dim=1)
        concat_e1 = torch.cat([e1a, e1b], dim=1)

        # Bottleneck
        b = self.bottleneck(torch.cat([p4a, p4b], dim=1))

        # Decoder part (shared across both images)
        d4 = self.decoder4(b, concat_e4)
        d3 = self.decoder3(d4, concat_e3)
        d2 = self.decoder2(d3, concat_e2)
        d1 = self.decoder1(d2, concat_e1)

        # Final output
        out = self.sigmoid(self.final_conv(d1))
        return out


# Model URL from DigitalOcean Spaces
MODEL_PATH = "siamese_unet.pth"

@st.cache_resource
def load_model():
    download_model()  # Ensure model is downloaded
    model = SiameseUNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# Check if model exists before loading
download_model()  # Only downloads if not already present
model = load_model()

# Streamlit UI
st.title("Change Detection using Siamese U-Net")
st.write("Upload two images: Pre-change and Post-change.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("Upload Pre-change Image", type=["png", "jpg", "jpeg"], key="pre")
with col2:
    uploaded_file2 = st.file_uploader("Upload Post-change Image", type=["png", "jpg", "jpeg"], key="post")

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1).convert("RGB")
    image2 = Image.open(uploaded_file2).convert("RGB")

    # Preprocess images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor1 = transform(image1).unsqueeze(0)  # Add batch dimension
    input_tensor2 = transform(image2).unsqueeze(0)

    with torch.no_grad():
        output_mask = model(input_tensor1, input_tensor2)
        output_mask = output_mask.squeeze().numpy()
        output_mask = (output_mask * 255).astype(np.uint8)

    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image1, caption="Pre-change Image", use_column_width=True)
    with col2:
        st.image(image2, caption="Post-change Image", use_column_width=True)
    with col3:
        st.image(output_mask, caption="Detected Changes", use_column_width=True)
