
### Prerequisites

#### 1. Environment Setup
Please use **Python 3.8.10** to install the dependencies:
```bash
# Create a new environment
conda create -n p2fusion python=3.8.10

# Activate the environment
conda activate p2fusion

# Install dependencies
pip install -r requirements.txt
```

#### 2. Model Weights
To run the model, you need to download the infrared teacher Segformer pre-trained weights. After downloading, please place the file in the nets/ directory.

Baidu Netdisk Link: https://pan.baidu.com/s/1GaSXNNPylVaiykVDJVr1qg?pwd=2kjc

Extraction Code: 2kjc

Ensure your directory structure looks like this:

P2Fusion/

├── nets/

│   └── best_epoch_weights.pth

#### 3.🚀 Training
The training process is highly customizable and supports both single-GPU and Multi-GPU Distributed Data Parallel (DDP) training. The configuration is entirely managed via a JSON file.

⚙️ Configuration 

Before starting the training, please configure the
```bash
 
 options/train_p2fusion.json 
 
```
 file. Ensure the dataset paths and hyperparameters are set correctly.

📂 Data Preparation

Before running the train, please ensure your dataset is organized in the following directory structure. 
The script expects strictly paired infrared and visible images with identical filenames.

Dataset/trainsets/
           
└── MSRS/     
    ├── ir/              
    └── vi/ 
    
    
💻 Basic Usage 

For standard training on a single GPU, run the following command:

```bash

python train.py --opt options/train_p2fusion.json

```
🏎️ Multi-GPU Training 

Our script natively supports PyTorch's Distributed Data Parallel (DDP) for faster training. To train the model across multiple GPUs (e.g., 2 GPUs), use torchrun:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --opt options/train_p2fusion.json 

```
Note: Replace 0,1 with your specific GPU IDs and --nproc_per_node with the number of GPUs you wish to use.

#### 4.🚀 Testing / Inference

This section provides instructions on how to run the inference script to evaluate the pre-trained P2Fusion model on your datasets.

📂 Data Preparation

Before running the test, please ensure your dataset is organized in the following directory structure. 
The script expects strictly paired infrared and visible images with identical filenames.

Dataset/valsets/
           
└── MSRS/     
    ├── ir/              
    └── vi/                
    
💻 Basic Usage
To run the evaluation on the default dataset (MSRS), simply execute:

```bash
python test.py \
    --dataset MSRS \
    --model_path ./Model/Infrared_Visible_Fusion/P2Fusion/models/best_E.pth

```
If you want to test on a different dataset (e.g., FMB) with custom folder names, use:

```bash
python test.py \
    --dataset FMB \
    --A_dir ir_img \
    --B_dir vis_img \
    --model_path /path/to/your/custom_model.pth
```