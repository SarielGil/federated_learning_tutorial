# Federated Learning: Chest X-ray Optimization POC

This project implements and compares several optimization techniques for Federated Learning (FL) using **NVIDIA FLARE** on a Chest X-ray classification task.

## 🚀 Experiments & Results

We compared 4 different configurations to find the optimal balance between accuracy, training speed, and communication efficiency:

| Configuration | Final Accuracy | Training Speed | Comm. Load | Stability |
| :--- | :--- | :--- | :--- | :--- |
| **Full FL (FP32)** | 87.5% | 1.0x (Baseline) | High (100%) | **Highest** |
| **Full FL (FP16/AMP)** | 90.0% | **2.5x Faster** | High (100%) | Medium |
| **LoRA (FP32)** | 50.0%* | 1.1x Faster | **Lowest (<2%)** | Low |
| **LoRA (FP16/AMP)**| **90.0%** | **~2.8x Faster** | **Lowest (<2%)** | **High** |

*\*Stuck at random guessing without hyperparameter optimization.*

---

## 💡 Lessons Learned & Best Practices

Throughout this development, we encountered and solved several critical Deep Learning and Federated Learning challenges:

### 1. The "Blind Model" Architecture
*   **Problem**: The initial model was stuck at 50% accuracy even with full training.
*   **Lesson**: Linear layers without non-linear activations (**ReLU**) suffer from "Linear Collapse"—mathematically reducing multiple layers to just one. 
*   **Fix**: Adding **ReLU** and **Dropout (0.5)** restored the model's ability to learn complex spatial features and prevented overfitting.

### 2. Optimizer Selection: Adam is King
*   **Problem**: standard SGD was too slow for the complex textures in X-rays.
*   **Lesson**: **Adam** provides much faster convergence for CV tasks in a Federated setting. However, we learned to **re-initialize the optimizer every round** to prevent stale momentum from destabilizing the global model after aggregation.

### 3. LoRA Needs "Force"
*   **Problem**: LoRA initially failed to learn at the same rate as the full model.
*   **Lesson**: Because LoRA only updates ~1% of parameters, standard learning rates are too weak.
*   **Fix**: Increasing the **Learning Rate (2e-3)** and doubling the **Local Epochs (2 per round)** allowed LoRA to achieve **90% accuracy**, matching the full model while being vastly more efficient.

### 4. FP16/AMP: Speed vs Stability
*   **Problem**: Full precision (FP32) is very slow, but mixed precision (FP16) can cause accuracy oscillations.
*   **Lesson**: Mixed precision training provides a **2-3x speedup**, but require slightly **finer LR tuning** (0.0003 instead of 0.0005) to maintain the same stability as FP32.

### 5. Pre-training is Essential
*   **Problem**: FL rounds starting from random weights often diverge.
*   **Lesson**: Pre-training the backbone on a small data subset (even just 100-200 samples) provided a "warm start" that stabilized the global model from Round 1.

---

## 🛠️ Project Structure
*   `federated_learning_tutorial_v7.ipynb`: Master experiment notebook.
*   `client_xray.py`: The FL training script with optimized hyperparameters.
*   `model.py`: The `ConvNet2` architecture and `LoRA` wrapper.
*   `diagnostic.py`: Tool used to verify parameter freezing and gradient flow.

---
*Developed for the Google DeepMind Advanced Agentic Coding Tutorial.*
