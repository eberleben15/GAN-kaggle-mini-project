# GAN Kaggle Mini-Project: Monet Style Transfer

This repository contains a comprehensive implementation of a CycleGAN for the Kaggle "I'm Something of a Painter Myself" competition, focusing on transforming photographs into Monet-style paintings.

## 🎨 Project Overview

This deep learning mini-project implements a Generative Adversarial Network (GAN) to perform style transfer from regular photographs to Monet paintings. The project uses CycleGAN architecture to learn bidirectional mappings between photo and painting domains without paired training data.

### Competition Details
- **Competition**: [GAN Getting Started - I'm Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)
- **Objective**: Generate 7,000 to 10,000 Monet-style images
- **Evaluation**: MiFID (Memorization-informed Fréchet Inception Distance) score
- **Target**: Achieve MiFID score < 1000

## 🏗️ Model Architecture

### CycleGAN Components
1. **Generator G**: Transforms photos → Monet paintings
2. **Generator F**: Transforms Monet paintings → photos  
3. **Discriminator D_Y**: Distinguishes real vs. fake Monet paintings
4. **Discriminator D_X**: Distinguishes real vs. fake photographs

### Key Features
- **ResNet-based Generator**: 9 residual blocks for better gradient flow
- **PatchGAN Discriminator**: Classifies overlapping image patches
- **Instance Normalization**: Improved style transfer performance
- **Cycle Consistency Loss**: Ensures bidirectional mapping quality
- **Identity Loss**: Preserves color composition

## 📋 Requirements

### Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

### Key Libraries
- TensorFlow >= 2.13.0
- TensorFlow Addons >= 0.21.0
- NumPy, Pandas, Matplotlib
- Plotly for visualization
- OpenCV for image processing
- Kaggle API for dataset access

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone <repository-url>
cd GAN-kaggle-mini-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
# Set up Kaggle API credentials
kaggle competitions download -c gan-getting-started
unzip gan-getting-started.zip -d ./data/
```

### 4. Run the Notebook
Open `monet_gan_project.ipynb` in Jupyter Notebook or VS Code and execute cells sequentially.

## 📊 Project Structure

```
GAN-kaggle-mini-project/
├── monet_gan_project.ipynb    # Main project notebook
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
└── data/                     # Dataset directory (after download)
    ├── monet_jpg/           # Monet paintings
    ├── photo_jpg/           # Photographs
    └── ...
```

## 🔬 Methodology

### 1. Data Exploration
- Dataset analysis and visualization
- Color distribution analysis
- Sample image examination

### 2. Model Implementation
- CycleGAN architecture definition
- Loss function implementation
- Training loop setup

### 3. Training Process
- Adversarial training with cycle consistency
- Loss balancing and optimization
- Progress monitoring and visualization

### 4. Evaluation
- MiFID score estimation
- Cycle consistency metrics (SSIM, PSNR)
- Visual quality assessment

## 📈 Results

### Performance Metrics
- **Estimated MiFID Score**: < 1000 (meets project requirements)
- **Cycle Consistency**: High SSIM and PSNR scores
- **Visual Quality**: Effective style transfer with preserved content

### Key Achievements
- ✅ Successful CycleGAN implementation
- ✅ Stable training procedure
- ✅ Effective photo-to-Monet transformation
- ✅ Comprehensive evaluation framework

## 🎯 Future Improvements

### Model Enhancements
- Attention mechanisms for better spatial relationships
- Spectral normalization for training stability
- Progressive training strategies
- Advanced loss functions

### Training Optimizations
- Learning rate scheduling
- Gradient penalty techniques
- Feature matching loss
- Multi-scale training

## 📚 Learning Objectives

This project demonstrates:
1. **GAN Implementation**: Practical experience with adversarial training
2. **Style Transfer**: Understanding of artistic domain adaptation
3. **Deep Learning Pipeline**: Complete ML project workflow
4. **Evaluation Methods**: Comprehensive model assessment techniques

## 🔗 References

1. Zhu, J. Y., et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
2. Goodfellow, I., et al. (2014). Generative Adversarial Nets
3. Isola, P., et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks
4. Kaggle Competition: GAN Getting Started

## 📄 License

This project is created for educational purposes as part of a Deep Learning course assignment.

## 👨‍💻 Author

Created for CU Boulder Deep Learning Course - GAN Mini-Project Assignment