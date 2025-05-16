import os
import sys

project_root = '/home/victorh/projects/gtx'
os.chdir(project_root)
sys.path.insert(0, project_root)

import logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime

from model.unet import UnetModel
from dataset import FluorescenceDataset
from utils.preprocess import load_data


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for fluorescence imaging model.")

    # General hyperparameters
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--nF', type=int, default=6, help='Number of fluroescent spatial frequencies (fluorescent images)')
    parser.add_argument('--learningRate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--xX', type=int, default=101, help='Image width')
    parser.add_argument('--yY', type=int, default=101, help='Image height')
    parser.add_argument('--decayRate', type=float, default=0.3, help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    # Scaling parameters
    parser.add_argument('--scaleFL', type=float, default=10e4, help='Scaling factor for fluorescence')
    parser.add_argument('--scaleOP0', type=float, default=10, help='Scaling for absorption coefficient (μa)')
    parser.add_argument('--scaleOP1', type=float, default=1, help='Scaling for scattering coefficient (μs\')')
    parser.add_argument('--scaleDF', type=float, default=1, help='Scaling for depth')
    parser.add_argument('--scaleQF', type=float, default=1, help='Scaling for fluorophore concentration')
    parser.add_argument('--scaleRE', type=float, default=1, help='Scaling for reflectance (optional)')

    # 3D Conv parameters
    parser.add_argument('--nFilters3D', type=int, default=128)
    parser.add_argument('--kernelConv3D', type=int, nargs=3, default=[3,3,3])
    parser.add_argument('--strideConv3D', type=int, nargs=3, default=[1,1,1])

    # 2D Conv parameters
    parser.add_argument('--nFilters2D', type=int, default=128)
    parser.add_argument('--kernelConv2D', type=int, nargs=2, default=[3,3])
    parser.add_argument('--strideConv2D', type=int, nargs=2, default=[1,1])

    # Data path
    parser.add_argument('--data_path', type=str, default='data/')

    return parser

def train(params):

    # Initialize model
    model = UnetModel(params)
    model = torch.compile(model).to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learningRate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params['decayRate'], patience=5)
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    
    # Init Dataset
    scale_params = {
        'fluorescence': params['scaleFL'],
        'mu_a': params['scaleOP0'],
        'mu_s': params['scaleOP1'],
        'depth': params['scaleDF'],
        'concentration_fluor': params['scaleQF'],
        'reflectance': params['scaleRE']
    }

    # Load data
    data = load_data(params['data_path'], scale_params)
    dataset = FluorescenceDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Dataset split into train and validation sets")

    # Init DataLoader
    train_loader = DataLoader(train_dataset, batch_size=params['batch'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=params['batch'], shuffle=False, num_workers=4)

    best_val_loss = float('inf')
    patience_counter = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'code_py/ckpt/{timestamp}/loss_{timestamp}.csv'
    os.makedirs(f'code_py/ckpt/{timestamp}', exist_ok=True)

    # Training loop
    for epoch in range(params['epochs']):
        print(f"Epoch {epoch + 1}/{params['epochs']}")
        model.train()
        train_loss = []
        qf_loss = []
        depth_loss = []
        for batch_idx, (fluorescence, mu_a, mu_s, concentration_fluor, depth) in enumerate(train_loader):
            print(f'Batch {batch_idx} of {len(train_loader)}') if batch_idx % 10 == 0 else None
            fluorescence = fluorescence.permute(0,3,1,2).unsqueeze(1).to(DEVICE)
            op = torch.cat([mu_a.unsqueeze(1), mu_s.unsqueeze(1)], dim=1).to(DEVICE)
            concentration_fluor = concentration_fluor.unsqueeze(1).to(DEVICE)
            depth = depth.unsqueeze(1).to(DEVICE)

            if epoch == 0 and batch_idx == 0:
                print(fluorescence.shape, op.shape, concentration_fluor.shape, depth.shape)

            optimizer.zero_grad()

            pred_qf, pred_depth = model(op, fluorescence)
            loss_qf = mae_loss(pred_qf, concentration_fluor)
            loss_depth = mae_loss(pred_depth, depth)
            loss = loss_qf + loss_depth
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            qf_loss.append(loss_qf.item())
            depth_loss.append(loss_depth.item())

        # Validation loop
        model.eval()
        val_loss = []
        val_qf_loss = []
        val_depth_loss = []
        with torch.no_grad():
            for batch_idx, (fluorescence, mu_a, mu_s, concentration_fluor, depth) in enumerate(val_loader):
                fluorescence, mu_a, mu_s, concentration_fluor, depth = fluorescence.to(DEVICE), mu_a.to(DEVICE), mu_s.to(DEVICE), concentration_fluor.to(DEVICE), depth.to(DEVICE)
                fluorescence = fluorescence.permute(0,3,1,2).unsqueeze(1).to(DEVICE)
                op = torch.cat([mu_a.unsqueeze(1), mu_s.unsqueeze(1)], dim=1).to(DEVICE)
                concentration_fluor = concentration_fluor.unsqueeze(1).to(DEVICE)
                depth = depth.unsqueeze(1).to(DEVICE)

                pred_qf, pred_depth = model(op, fluorescence)
                loss_qf = mae_loss(pred_qf, concentration_fluor)
                loss_depth = mae_loss(pred_depth, depth)
                loss = loss_qf + loss_depth
                val_loss.append(loss.item())
                val_qf_loss.append(loss_qf.item())
                val_depth_loss.append(loss_depth.item())

        # Print loss
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        logger.info(f"Epoch {epoch + 1}/{params['epochs']}, Train Loss: {avg_train_loss:.4f}, qf_loss: {np.mean(qf_loss):.4f}, depth_loss: {np.mean(depth_loss):.4f}, val_loss: {avg_val_loss:.4f}, val_qf_loss: {np.mean(val_qf_loss):.4f}, val_depth_loss: {np.mean(val_depth_loss):.4f}")

        # write to csv file
        write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0

        with open(log_file, 'a') as f:
            if write_header:
                f.write("Epoch,Train_Loss,QF_Loss,Depth_Loss,Val_Loss,Val_QF_Loss,Val_Depth_Loss\n")
            f.write(f"{epoch + 1},{avg_train_loss:.4f},{np.mean(qf_loss):.4f},{np.mean(depth_loss):.4f},{avg_val_loss:.4f},{np.mean(val_qf_loss):.4f},{np.mean(val_depth_loss):.4f}\n")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'code_py/ckpt/{timestamp}/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        
if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args)

    np.random.seed(1024)
    torch.manual_seed(1024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1024)
    torch.backends.cudnn.deterministic = True

    try:
        train(params)
    except KeyboardInterrupt:
        print("Training interrupted by user")
