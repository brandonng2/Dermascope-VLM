# Dermascope-VLM
 
## Installation & Setup
 
Follow these steps to set up the environment and get the HAM10000 dataset ready for training.
 
### 1. Clone the repository
 
```bash
git clone https://github.com/yourusername/Dermascope-VLM.git
cd Dermascope-VLM
```
 
### 2. Set up Kaggle API credentials
 
The dataset is hosted on Kaggle, so you need a Kaggle account and API key:
 
1. Go to **Kaggle Account → API** and create a new API token.
2. You can either:
   - Place `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/Mac) and run:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Or set environment variables in your shell:
     ```bash
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_api_key
     ```
 
This ensures the Kaggle CLI can authenticate and download the dataset.
 
### 3. Create Conda environment
 
This project uses Conda to manage dependencies. The required packages are listed in `environment.yml`.
 
```bash
# Create environment from YAML file
conda env create -f environment.yml
 
# Activate the environment
conda activate derma
```
 
### 4. Download HAM10000 dataset
 
We provide a script to automatically download and unzip the dataset:
 
```bash
bash scripts/download_data.sh
```
 
- This will download the dataset and unzip it into the `data/` folder.
- You do not need to manually run `chmod +x` if using `bash scripts/download_data.sh`.
 
After running this, you should see:
 
```
data/
├── images/
├── masks/        # if segmentation task
└── metadata.csv
```