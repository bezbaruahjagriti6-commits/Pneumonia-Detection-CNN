# Pneumonia Detection from Chest X-Rays | Custom PyTorch CNN

##  Project Overview
This project implements a high-performance deep learning pipeline to automate the detection of Pneumonia from chest X-ray images. Built from scratch using a custom **Convolutional Neural Network (CNN)** in PyTorch, the model is specifically optimized for clinical sensitivity, ensuring that the vast majority of positive cases are identified for physician review.

##  Performance Summary
| Metric | Result |
| :--- | :--- |
| **Peak Validation Accuracy** | 97.63% |
| **Final Test Accuracy** | 80.93% |
| **Sensitivity (Recall)** | **99.2%** |
| **Hardware Acceleration** | NVIDIA RTX 4050 (CUDA) |

### The "Clinical Success" Factor
In medical AI and bioinformatics, a **False Negative** (missing a sick patient) is far more critical than a **False Positive** (additional testing for a healthy patient). My model achieved a near-perfect recall rate, catching **387 out of 390** pneumonia cases in a completely unseen, difficult test set.

##  Data Engineering & Pipeline
### 1. Solving the "16-Image" Validation Problem
The raw Kaggle dataset provided a severely unbalanced validation set of only 16 images, which leads to unstable training metrics. I engineered a programmatic solution by:
- Merging the original training and validation directories via a custom Python script.
- Executing a **Stratified 80/20 Split** to create a statistically robust validation set of ~1,000 images.
- This allowed for accurate **Checkpointing**, saving the model at its true peak (Epoch 6) before overfitting occurred.

### 2. Advanced Architecture
- **CNN Backbone:** Custom multi-layer feature extractor using PyTorch.
- **Squeeze-and-Excitation (SE) Blocks:** Added an attention mechanism that re-weights feature channels, helping the AI focus on lung opacities while ignoring background noise (ribs, skin, etc.).
- **Class Weighting:** Applied a penalty of `0.3474` to the loss function to account for the class imbalance (3,106 Pneumonia vs. 1,079 Normal samples) during training.

##  Confusion Matrix Analysis
The following matrix illustrates the model's cautious, highly sensitive performance on the 624-image unseen test set:

## [Confusion Matrix](presentation_confusion_matrix.png)

- **True Positives:** 387 (Sick patients correctly identified)
- **True Negatives:** 118 (Healthy patients correctly identified)
- **False Positives:** 116 (Healthy patients flagged for review - safe clinical error)
- **False Negatives:** 3 (Sick patients missed - near-zero critical error rate)
- 
## 🚀 How to Use This Project

### 1. Clone the Repository
## Open your terminal and download the code to your local machine:
```bash
git clone [https://github.com/bezbaruahjagriti6-commits/Pneumonia-Detection-CNN.git](https://github.com/bezbaruahjagriti6-commits/Pneumonia-Detection-CNN.git)
cd Pneumonia-Detection-CNN

## 2. Install Dependencies
It is recommended to use a virtual environment. Install the required Python packages using:
Bash
pip install -r requirements.txt

## 3. Run the Pipeline
First, split the raw data into proper training and validation sets:
Bash
python prepare_data.py

## 4. Next, run the model training and evaluation script:
Bash
python imageanalysis.py

## Technologies & Skills Used
Language: Python

Framework: PyTorch (Torchvision)

Data Handling: Pandas, Scikit-learn

Visualization: Matplotlib, Seaborn

Focus Areas: Deep Learning, Genomic Data Workflows, Medical Image Analysis 

# Conclusion
In conclusion, this project was not just about achieving a high overall accuracy, but about engineering a model that understands the high stakes of medical diagnostics.
By correcting a fundamentally flawed dataset and prioritizing clinical safety, this custom PyTorch pipeline achieved a 99.2% Sensitivity rate. By missing only 3 out of 390 actual pneumonia cases in unseen data, the model demonstrates the 'cautious' diagnostic behavior required for real-world healthcare—proving my ability to build highly accurate, statistically rigorous, and clinically viable AI.
