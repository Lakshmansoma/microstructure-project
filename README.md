# Microstructure Evolution Modeling with 3D CNN Autoencoders and Grainwise LSTM Networks

This project builds a **novel deep learning system** for analyzing, reconstructing, and predicting microstructure evolution in FCC materials.  
Unlike traditional microstructure ML projects that use only static grain features or 2D slices, this work combines:

- **3D voxel-level morphology learning** with a CNN autoencoder  
- **Temporal grainwise evolution modeling** with an LSTM sequence predictor  

This dual-model pipeline delivers deeper physical insights and superior predictive capability.

---

# 🚀 Project Overview

The project consists of **two major components**, each targeting a different aspect of microstructure behavior.

---

## 🧩 1. 3D CNN Autoencoder — Voxelwise Morphology Learning

We train a **3D Convolutional Autoencoder** on voxelized “DistanceFrom” fields extracted from FCC samples.  
The autoencoder learns:

- 3D grain boundary morphology  
- Spatial heterogeneity  
- Local deformation patterns  
- Compressed latent representations of microstructure state  

### Features:
- Full 3D reconstruction capability  
- Latent-space embedding useful for visualization or downstream ML tasks  
- Handles all voxel HDF5 datasets under `FCC_voxelwise`  

### Train the model:
```bash
python -m src.training.train_autoencoder
