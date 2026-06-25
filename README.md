
🚀 Training
The training process is highly customizable and supports both single-GPU and Multi-GPU Distributed Data Parallel (DDP) training. The configuration is entirely managed via a JSON file.

⚙️ Configuration 
Before starting the training, please configure the
 
 options/train_p2fusion.json 
 
 file. Ensure the dataset paths and hyperparameters are set correctly.

💻 Basic Usage 
For standard training on a single GPU, run the following command:

python train.py --opt options/train_p2fusion.json

🏎️ Multi-GPU Training 
Our script natively supports PyTorch's Distributed Data Parallel (DDP) for faster training. To train the model across multiple GPUs (e.g., 2 GPUs), use torchrun:

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --opt options/train_p2fusion.json \
    --launcher pytorch
Note: Replace 0,1 with your specific GPU IDs and --nproc_per_node with the number of GPUs you wish to use.

🚀 Testing / Inference
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

python test.py \
    --dataset MSRS \
    --model_path ./Model/Infrared_Visible_Fusion/P2Fusion/models/best_E.pth
If you want to test on a different dataset (e.g., FMB) with custom folder names, use:

python test.py \
    --dataset FMB \
    --A_dir ir_img \
    --B_dir vis_img \
    --model_path /path/to/your/custom_model.pth
