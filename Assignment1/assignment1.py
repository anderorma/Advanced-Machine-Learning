import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.init as init

# Check if an NVIDIA GPU (CUDA) is available. 
# If yes, we use 'cuda'; otherwise, we fall back to the 'cpu'.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device choice so we know if the code is running on the GPU or CPU
print(f"Using device: {DEVICE}\n")

stations = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", 
    "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan", 
    "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"
]

df_list = [pd.read_csv(f"Assignment1/Datasets/PRSA_Data_{name}_20130301-20170228.csv") for name in stations]

beijing_total = pd.concat(df_list, ignore_index=True)
print(beijing_total.head())
print(beijing_total.tail())
