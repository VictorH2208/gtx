import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model.unet import UnetModel
from dataset import FluorescenceDataset
from preprocess import load_data


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
    model.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learningRate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params['decayRate'], patience=5)
    mse_loss = torch.nn.MSELoss()
    
    # Init Dataset
    scale_params = {
        'fluorescence': params['scaleFL'],
        'mu_a': params['scaleOP0'],
        'mu_s': params['scaleOP1'],
        'depth': params['scaleDF'],
        'concentration_fluor': params['scaleQF'],
        'reflectance': params['scaleRE']
    }
    data = load_data(params['data_path'], scale_params)
    dataset = FluorescenceDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Dataset split into train and validation sets")

    # Init DataLoader
    train_loader = DataLoader(train_dataset, batch_size=params['batch'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch'], shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in tqdm(range(params['epochs']), desc="Training"):
        print(f"Epoch {epoch + 1}/{params['epochs']}")
        model.train()
        train_loss = []
        for batch_idx, (fluorescence, mu_a, mu_s, concentration_fluor, depth) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}/{len(train_loader)}")
            fluorescence, mu_a, mu_s, concentration_fluor, depth = fluorescence.to(DEVICE), mu_a.to(DEVICE), mu_s.to(DEVICE), concentration_fluor.to(DEVICE), depth.to(DEVICE)
            fluorescence = fluorescence.unsqueeze(1)
            op = torch.cat([mu_a, mu_s], dim=1)

            optimizer.zero_grad()

            pred_qf, pred_depth = model(op, fluorescence)
            loss_qf = mse_loss(pred_qf, concentration_fluor)
            loss_depth = mse_loss(pred_depth, depth)
            loss = loss_qf + loss_depth
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # Validation loop
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_idx, (fluorescence, mu_a, mu_s, concentration_fluor, depth) in enumerate(val_loader):
                print(f"Batch {batch_idx + 1}/{len(val_loader)}")
                fluorescence, mu_a, mu_s, concentration_fluor, depth = fluorescence.to(DEVICE), mu_a.to(DEVICE), mu_s.to(DEVICE), concentration_fluor.to(DEVICE), depth.to(DEVICE)
                op = torch.cat([mu_a, mu_s], dim=1)
                pred_qf, pred_depth = model(op, fluorescence)
                loss_qf = mse_loss(pred_qf, concentration_fluor)
                loss_depth = mse_loss(pred_depth, depth)
                loss = loss_qf + loss_depth
                val_loss.append(loss.item())

        # Print loss
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch + 1}/{params['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
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
    train(params)