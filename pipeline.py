import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import physics_tools as pt
from materials import Silicon, q, k_B, m_0, GalliumArsenide


# Create the Neural Network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        return self.output(x)


# generate data

def generate_training_data(filename="Silicon_fermi_dataset.csv", grid_size=100):
    if os.path.exists(filename):
        print(f"[*] Dataset '{filename}' already exists. Skipping generation.")
        return filename

    print(f"[*] Generating new dataset...")
    temperatures = np.linspace(100, 1000, 100)
    doping_levels = np.logspace(10, 24, 100)
    data = []
    N_A = 0.0

    for T in temperatures:
        E_c_j = Silicon.get_bandgap_j(T)
        E_v_j = 0.0

        for N_D in doping_levels:
            # Ask the material for its masses
            m_n_eff = Silicon.me_eff_dos * m_0
            m_p_eff = Silicon.mh_eff_dos * m_0

            lower_search_bound = E_v_j - (0.1 * q)
            upper_search_bound = E_c_j + (0.1 * q)
            # The *args pass right through bisection_method into the objective function
            Ef_j = pt.bisection_method(
                pt.charge_neutrality_objective,
                lower_search_bound, upper_search_bound,  # The bounds for bisection
                T, N_A, N_D, E_c_j, E_v_j, m_n_eff, m_p_eff,  # The *args
                tol=1e-6 * q
            )

            if Ef_j is None:
                print(f"[!] Warning: Solver failed at T={T}K, N_D={N_D:.1e}. Skipping point.")
                continue

            data.append({
                "Temperature_K": T,
                "Log10_N_D": np.log10(N_D),
                "Fermi_Level_eV": Ef_j / q,
            })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"[*] Saved {len(df)} records to '{filename}'.")
    return filename

# Train Model

def train_model(data_filename, epochs=2000, lr=0.005):
    """

    :param data_filename:
    :param epochs:
    :param lr:
    :return:
    """
    print(f"\n[*] Loading data from '{data_filename}'...")
    df = pd.read_csv(data_filename)

    # Extract inputs (X) and outputs (Y)
    # X needs to have 2 columns: [Temperature, Log_Doping]
    X_raw = df[['Temperature_K', 'Log10_N_D']].values
    Y_raw = df[['Fermi_Level_eV']].values

    # The Golden Rule: Scale the inputs to be between 0 and 1
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_scaled = (X_raw - X_min) / (X_max - X_min)

    # Convert NumPy arrays to PyTorch Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.float32)

    model = Net()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[*] Training model...")
    for epoch in range(epochs):
        predictions = model(X_tensor)
        loss = loss_fn(predictions, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:4d} | MSE Loss: {loss.item():.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'X_min': X_min,
        'X_max': X_max
    }, "Silicon_fermi_surrogate.pth")

    print("[*] Training Complete! Model and scalers saved to 'Silicon_fermi_surrogate.pth'")
    return model, X_min, X_max

# Execute

if __name__ == "__main__":
    print("=== SEMICONDUCTOR SURROGATE ML PIPELINE ===")

    dataset_file = generate_training_data()

    trained_model, x_min, x_max = train_model(dataset_file, epochs=2000)

    print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")































