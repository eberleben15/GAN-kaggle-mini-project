#!/usr/bin/env python3
"""
Data Download and Preprocessing Script for Monet GAN Project
"""

import os
import zipfile
import kaggle
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def setup_kaggle_api():
    """Setup Kaggle API credentials."""
    print("Setting up Kaggle API...")
    
    # Check if kaggle.json exists
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("⚠️  Kaggle API credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle API credentials found!")
    return True

def download_dataset():
    """Download the Monet GAN dataset from Kaggle."""
    if not setup_kaggle_api():
        return False
    
    print("Downloading dataset from Kaggle...")
    
    try:
        # Create data directory
        data_dir = Path('./data')
        data_dir.mkdir(exist_ok=True)
        
        # Download competition data
        kaggle.api.competition_download_files(
            'gan-getting-started', 
            path=str(data_dir), 
            quiet=False
        )
        
        # Extract zip file
        zip_path = data_dir / 'gan-getting-started.zip'
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file
            zip_path.unlink()
            print("✅ Dataset downloaded and extracted successfully!")
            return True
        else:
            print("❌ Download failed - zip file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def verify_dataset():
    """Verify the downloaded dataset structure."""
    print("\nVerifying dataset structure...")
    
    data_dir = Path('./data')
    required_dirs = ['monet_jpg', 'photo_jpg']
    
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob('*.jpg')))
            print(f"✅ {dir_name}: {file_count} images")
        else:
            print(f"❌ {dir_name}: Directory not found")
            return False
    
    return True

def create_sample_visualization():
    """Create a sample visualization of the dataset."""
    print("\nCreating sample visualization...")
    
    data_dir = Path('./data')
    monet_dir = data_dir / 'monet_jpg'
    photo_dir = data_dir / 'photo_jpg'
    
    if not (monet_dir.exists() and photo_dir.exists()):
        print("❌ Dataset directories not found")
        return
    
    # Get sample images
    monet_files = list(monet_dir.glob('*.jpg'))[:5]
    photo_files = list(photo_dir.glob('*.jpg'))[:5]
    
    if len(monet_files) == 0 or len(photo_files) == 0:
        print("❌ No sample images found")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Display Monet samples
    for i, img_path in enumerate(monet_files):
        img = Image.open(img_path)
        img = img.resize((256, 256))
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Monet {i+1}')
        axes[0, i].axis('off')
    
    # Display Photo samples
    for i, img_path in enumerate(photo_files):
        img = Image.open(img_path)
        img = img.resize((256, 256))
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Photo {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Dataset Sample Images', fontsize=16)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Sample visualization saved as 'dataset_samples.png'")

def main():
    """Main function to download and setup the dataset."""
    print("=" * 50)
    print("  MONET GAN PROJECT - DATA SETUP")
    print("=" * 50)
    
    # Download dataset
    if download_dataset():
        # Verify dataset
        if verify_dataset():
            # Create sample visualization
            create_sample_visualization()
            
            print("\n" + "=" * 50)
            print("🎉 Dataset setup completed successfully!")
            print("You can now run the main notebook: monet_gan_project.ipynb")
            print("=" * 50)
        else:
            print("\n❌ Dataset verification failed")
    else:
        print("\n❌ Dataset download failed")
        print("Please check your Kaggle API setup and try again")

if __name__ == "__main__":
    main()
