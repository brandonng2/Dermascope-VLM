# DermaVision

Comprehensive skin disease classification on HAM10000 using CNNs, Transformers, and Vision-Language Models. Compares a custom CNN, fine-tuned ResNet-50, and fine-tuned Swin-T against zero-shot CLIP and DermLIP baselines. Includes sequential ablations over optimizer, learning rate, and class imbalance strategy for supervised models, plus Grad-CAM interpretability analysis.

## Project Overview

HAM10000 is a 10,015-image dermatoscopy dataset spanning 7 classes (melanoma, melanocytic nevi, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, vascular lesions). The dataset is heavily imbalanced — melanocytic nevi account for ~67% of samples — making class imbalance handling a key design consideration.

This project trains and evaluates five models with comprehensive analysis:

| Model         | Type        | Training       |
| ------------- | ----------- | -------------- |
| Custom CNN    | CNN         | From scratch   |
| ResNet-50     | CNN         | Fine-tuned     |
| Swin-T Tiny   | Transformer | Fine-tuned     |
| CLIP ViT-B/32 | VLM         | Zero-shot only |
| DermLIP       | Medical VLM | Zero-shot only |

Each supervised model goes through a sequential ablation (optimizer → learning rate → imbalance strategy) before a final best-config training run. Results are evaluated on a held-out test set using accuracy, macro AUC, micro AUC, and per-class AUC.

## Project Structure

```
Dermascope-VLM/
├── configs/                     # Configuration files for all models and evaluations
│   ├── ablation_notebooks.json              # Notebook configuration settings
│   ├── cnn.json                             # CNN architecture and training hyperparameters
│   ├── dataset.json                         # Dataset configuration (splits, transforms, paths)
│   ├── eval_gradcam.json                    # Grad-CAM layer selection and parameters
│   ├── eval_vlm.json                        # VLM evaluation config (prompt templates, models)
│   ├── resnet50.json                        # ResNet-50 configuration and hyperparameters
│   └── swin_t.json                          # Swin-T configuration and hyperparameters
│
├── data/
│   ├── images/                  # HAM10000 images (.jpg)
│   ├── masks/                   # HAM10000 masks (.jpg)
│   └── GroundTruth.csv          # Class labels for all images
│
├── notebooks/
│   ├── ablation_cnn.ipynb                    # CNN ablation analysis and best config selection
│   ├── ablation_resnet50.ipynb               # ResNet-50 ablation analysis
│   ├── ablation_swin_t.ipynb                 # Swin-T ablation analysis
│   ├── dataset.ipynb                         # Data exploration and class distribution
│   ├── eval_gradcam.ipynb                    # Grad-CAM visualization and interpretability
│   ├── evaluate_all_models.ipynb             # Comprehensive evaluation across all 5 models
│   ├── evaluate_best_VLM_models.ipynb        # CLIP and DermLIP comparison
│   └── evaluate_best_supervised_models.ipynb # CNN, ResNet-50, Swin-T comparison
│
├── results/
│   ├── cnn/                                 # CNN results and checkpoints
│   │   ├── ablation/                        # Sequential ablation runs across 3 axes
│   │   │   ├── imbalance/                   # Class imbalance strategy sweep (none, sampler, weighted)
│   │   │   ├── lr/                          # Learning rate sweep (1e-6, 1e-5, 0.0001, 0.001)
│   │   │   ├── optimizer/                   # Optimizer sweep (adam, sgd)
│   │   │   └── summary.json                 # Best config summary
│   │   └── best_model/                      # Final training with best config
│   │       ├── best_cnn.pth                 # Model checkpoint
│   │       ├── cnn_classification_report.txt # Precision/recall/F1 per class
│   │       ├── cnn_confusion_matrix.png     # Visualization
│   │       ├── cnn_sample_predictions.png   # Example predictions
│   │       ├── cnn_training_curves.png      # Loss and accuracy curves
│   │       └── training_log.txt             # Epoch-by-epoch logs
│   │
│   ├── gradcam/                             # Grad-CAM interpretability analysis for all models
│   │   ├── clip/                            # CLIP Grad-CAM results with examples and metrics
│   │   │   ├── gradcam_examples.png         # Visual examples
│   │   │   └── gradcam_metrics.json         # Quantitative metrics
│   │   ├── cnn/                             # CNN Grad-CAM results (w/ png and json)
│   │   ├── dermlip/                         # DermLIP Grad-CAM results (w/ png and json)
│   │   ├── figures/                         # Comprehensive analysis figures
│   │   ├── resnet50/                        # ResNet-50 Grad-CAM results (w/ png and json)
│   │   ├── swint/                           # Swin-T Grad-CAM results(w/ png and json)
│   │   └── gradcam_summary.json             # Overall Grad-CAM analysis summary
│   │
│   ├── resnet50/                            # ResNet-50 results and checkpoints
│   │   ├── ablation/                        # Sequential ablation runs (same as CNN ablation)
│   │   │   ├── imbalance/
│   │   │   ├── lr/
│   │   │   ├── optimizer/
│   │   │   └── summary.json
│   │   └── best_model/                      # Best ResNet-50 checkpoint, metrics, visualizations, logs
│   │
│   ├── swin_t/                              # Swin-T results and checkpoints
│   │   ├── ablation/                        # Sequential ablation runs (same as CNN ablation)
│   │   │   ├── imbalance/
│   │   │   ├── lr/
│   │   │   ├── optimizer/
│   │   │   └── summary.json
│   │   └── best_model/                      # Best Swin-T checkpoint, metrics, visualizations, logs
│   │
│   └── vlm/                                 # Vision-Language Model zero-shot results
│       ├── clip/                            # CLIP ViT-B/32 results with different prompts
│       │   ├── clinical/                    # Results with "clinical" prompt template
│       │   ├── dermatoscopic/               # Results with "dermatoscopic" prompt template
│       │   ├── dermoscopy/                  # Results with "dermoscopy" prompt template
│       │   ├── label_only/                  # Results with "label only" prompt template
│       │   ├── photo_of/                    # Results with "photo of" prompt template
│       │   ├── skin_lesion/                 # Results with "skin lesion" prompt template
│       │   ├── this_is/                     # Results with "this is" prompt template
│       │   └── summary.json                 # Best prompt summary
│       └── dermlip/                         # DermLIP Medical VLM results
│           ├── clinical/
│           ├── dermatoscopic/
│           ├── dermoscopy/
│           ├── label_only/
│           ├── photo_of/
│           ├── skin_lesion/
│           ├── this_is/
│           └── summary.json
│
├── scripts/
│   └── download_data.sh                      # Downloads and unzips HAM10000 from Kaggle
│
├── src/
│   ├── cnn/                                  # Custom CNN model training and ablation
│   │   ├── ablation_cnn.py                  # Sequential ablation: optimizer → LR → imbalance
│   │   ├── custom_cnn.py                    # 4-block CNN architecture definition
│   │   └── train_cnn.py                     # Final training with best config
│   ├── resnet50/                            # Fine-tuned ResNet-50 model with model, ablation, training
│   ├── swin_t/                              # Swin-T Tiny model with model, ablation, training
│   ├── dataset.py                           # HAM10000Dataset, data loaders, transforms, class weights
│   ├── eval_gradcam.py                      # Grad-CAM generation for all models
│   ├── eval_vlm.py                          # Zero-shot VLM evaluation (CLIP, DermLIP with prompt templates)
│   ├── train.py                             # Shared train/eval loop for all supervised models
│   └── utils.py                             # Metrics computation, plotting, Grad-CAM utilities
│
├── environment.yml
└── README.md
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Dermascope-VLM.git
cd Dermascope-VLM
```

### 2. Set up Kaggle API credentials

The dataset is hosted on Kaggle, so you need a Kaggle account and API key:

1. Go to **Kaggle Account → API** and create a new API token.
2. You can either:
   - Place `kaggle.json` in `~/.kaggle/kaggle.json` and run:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Or set environment variables:
     ```bash
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_api_key
     ```

### 3. Create Conda environment

```bash
conda env create -f environment.yml
conda activate derma
```

### 4. Download HAM10000 dataset

```bash
bash scripts/download_data.sh
```

After running this, `data/` should contain:

```
data/
├── images/
├── masks/
└── GroundTruth.csv
```

## Running

### 1. Ablation (run first to find best hyperparameters)

Run sequential ablations for each supervised model across optimizer, learning rate, and imbalance strategy:

```bash
python src/cnn/ablation_cnn.py
python src/resnet50/ablation_resnet50.py
python src/swin_t/ablation_swin_t.py
```

Each ablation script writes a `summary.json` with the best config to `results/<model>/ablation/`. Review these results to select best hyperparameters before final training.

### 2. Final training (update best config in train script first)

After reviewing ablation results, update the hyperparameters in each training script and run:

```bash
python src/cnn/train_cnn.py
python src/resnet50/train_resnet50.py
python src/swin_t/train_swin_t.py
```

Generates checkpoint and classification report in `results/<model>/best_model/`.

### 3. VLM evaluation (zero-shot CLIP & DermLIP)

Evaluate pre-trained Vision-Language Models without fine-tuning:

```bash
python src/eval_vlm.py
```

Results are saved to `results/vlm/<model>/` with different prompt templates (clinical, dermatoscopic, dermoscopy, label_only, photo_of, skin_lesion, this_is). Check `eval_vlm.json` config for settings.

### 4. Grad-CAM interpretability analysis

Generate Grad-CAM visualizations for all models:

```bash
python src/eval_gradcam.py
```

Outputs visualization figures to `results/gradcam/figures/` and metrics to `results/gradcam/<model>/gradcam_metrics.json`. Configuration in `eval_gradcam.json`.

### 5. Notebook-based analysis

Open Jupyter to explore results interactively:

**Data Exploration:**

- `notebooks/dataset.ipynb` — HAM10000 dataset overview and class distribution analysis

**Supervised Model Evaluation:**

- `notebooks/evaluate_best_supervised_models.ipynb` — Compare CNN, ResNet-50, and Swin-T on test set
- `notebooks/ablation_cnn.ipynb` — Analyze CNN ablation results
- `notebooks/ablation_resnet50.ipynb` — Analyze ResNet-50 ablation results
- `notebooks/ablation_swin_t.ipynb` — Analyze Swin-T ablation results

**Vision-Language Model Evaluation:**

- `notebooks/evaluate_best_VLM_models.ipynb` — Compare CLIP and DermLIP across prompt templates

**Comprehensive Analysis:**

- `notebooks/evaluate_all_models.ipynb` — Compare all 5 models (3 supervised + 2 VLM) together
- `notebooks/eval_gradcam.ipynb` — Grad-CAM visualization and interpretability analysis

## Key Design Decisions

### Supervised Model Training

- `src/train.py` and `src/utils.py` are shared across all supervised models — one fix applies everywhere
- Val split is 10% of training data, stratified by class — val loss is tracked each epoch for early stopping and ablation selection
- Class weights computed from training set targets only, never from val or test
- Ablations are sequential: each axis fixes the best result from the previous axis before sweeping the next

### Vision-Language Model Evaluation

- CLIP and DermLIP are evaluated zero-shot using multiple prompt templates (e.g., "clinical description", "dermatoscopic view", etc.)
- Prompt variations explore how different class label framings affect VLM classification
- No fine-tuning or adaptation — pure zero-shot performance to establish strong non-trainable baselines

### Grad-CAM Interpretability

- Grad-CAM analysis applies to all 5 models (CNN, ResNet-50, Swin-T, CLIP, DermLIP)
- Visualizations highlight which image regions contribute most to model predictions
- Enables diagnosis of whether models focus on clinically relevant features vs. data artifacts

### Configuration Management

All model and evaluation settings are centralized in `configs/`:

- Per-model configs (cnn.json, resnet50.json, swin_t.json) specify architecture, optimizer, and hyperparameters
- eval_vlm.json controls prompt templates and VLM evaluation settings
- eval_gradcam.json controls Grad-CAM layer selection and visualization parameters
