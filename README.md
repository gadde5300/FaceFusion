# FaceFusion

**FaceFusion: Ethnic Feature Blending Diffusion Model**

---

## Overview

FaceFusion is an innovative machine learning project that generates synthetic facial images by blending features from different ethnic groups using a conditional diffusion model. The project aims to create realistic, diverse facial representations by combining characteristics from Oriental, Indian, and European ethnicities.

---

## Key Features

- 🔬 **Custom Class-Conditional UNet-based Diffusion Model**
- 🌍 **Multi-Ethnic Face Generation**
- 🎨 **Flexible Ethnic Feature Blending**
- 📊 **Controllable Image Synthesis**

---

## Project Description

The goal of FaceFusion is to generate new faces by intelligently combining features from three distinct ethnic groups:

- **Orientals**
- **Indians**
- **White Europeans**

### Capabilities

- Generate up to **10 unique face images** for each ethnic combination
- Adjust class proportions dynamically (e.g., 50% Oriental, 50% Indian)
- Create faces representing multiple ethnic groups simultaneously

---

## Technical Details

### Model Architecture
- **Custom UNet-based Diffusion Model**
- **Class Conditioning** using Learned Embeddings
- **RGB Image Output** (64×64 pixels)

### Data Preparation
- Sourced from open platforms: **Kaggle** and **Roboflow**
- Facial region extraction using **OpenCV**
- Preprocessed and normalized images

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/facefusion.git

# Install dependencies
pip install -r requirements.txt
