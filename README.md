# Federated Learning POC: Chest X-ray Classification

This project demonstrate a Federated Learning (FL) Proof of Concept (POC) using **NVIDIA FLARE** to train a deep CNN model (**ConvNet2**) on the Chest X-ray dataset. The implementation explores advanced optimization techniques like **LoRA (Low-Rank Adaptation)** and **Mixed Precision Training**.

## Key Features

- **Custom ConvNet2 Architecture**: A deep CNN designed for image classification, improved with ReLU activations and Dropout to ensure robust learning and prevent linear collapse.
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning that trains only a tiny fraction (~1%) of parameters, significantly reducing communication overhead.
- **Backbone Pre-training**: Stable initialization by pre-training the model backbone on a data subset before starting the federated rounds.
- **Comparative Analysis**: Built-in experiments to compare LoRA vs. Full fine-tuning in terms of accuracy, convergence speed, and execution time.
- **NVIDIA FLARE Integration**: Uses the `FedAvgRecipe` for seamless federated averaging simulation.
- **TensorBoard Tracking**: Real-time visualization of global and local training metrics.

## Project Structure

```bash
.
├── README.md               # Project documentation
├── chest_xray_poc.ipynb    # Main experiment notebook (LoRA vs. Full FL)
├── client_xray.py          # Client training logic for NVIDIA FLARE
├── model.py                # Model definition (ConvNet2 & LoRA wrapper)
├── requirements.txt        # Python dependencies
├── setup.sh                # Environment setup script
├── run_notebook.sh         # Script to launch the environment
└── .gitignore              # Git exclusion rules
```

## Getting Started

### 1. Requirements
Ensure you have Python 3.9+ installed.

### 2. Setup
Clone the repository and run the setup script to create a virtual environment and install dependencies:
```bash
./setup.sh
```

### 3. Run the Experiment
Load the virtual environment and launch the Jupyter notebook:
```bash
source venv/bin/activate
# Open chest_xray_poc.ipynb in your favorite editor/Jupyter instance
```

## Experiment Details

The `chest_xray_poc.ipynb` notebook walks through:
1. **Pre-training**: Initializing the backbone on a small subset of data.
2. **LoRA Experiment**: Running a 5-round FL simulation with LoRA adapters.
3. **Full FL Comparison**: Running an identical simulation with full-parameter updates.
4. **Summary**: A side-by-side comparison table of duration and performance.

## Metrics Compared
- **Accuracy**: Final test accuracy on the global model.
- **Convergence**: Loss curves per round.
- **Time**: Total wall-clock time for the simulation.
- **Efficiency**: Parameter count and communication cost.

---
*Created as part of the Federated Learning optimization tutorial.*
