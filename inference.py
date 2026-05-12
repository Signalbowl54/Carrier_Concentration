import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# We must import the architecture from your pipeline script
# so PyTorch knows how to reconstruct the network.
from pipeline import Net


class SiliconPhysicsEngine:
    def __init__(self, model_path="silicon_fermi_surrogate.pth"):
        print(f"[*] Booting AI Surrogate from {model_path}...")

        # 1. Load the dictionary containing our weights and scalers
        checkpoint = torch.load(model_path, weights_only=False)

        # 2. Rebuild the network architecture
        self.model = Net()

        # 3. Inject the trained "brain" (the weights) into the architecture
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 4. Lock the model! (Disables training features, speeds up inference)
        self.model.eval()

        # 5. Extract the scaling rules
        self.X_min = checkpoint['X_min']
        self.X_max = checkpoint['X_max']

    def predict_fermi_level(self, Temp_K, N_D_m3):
        """Instantly predicts the Fermi level without running any integrals."""

        # Step A: Format and scale the raw inputs exactly like we did in training
        raw_input = np.array([[Temp_K, np.log10(N_D_m3)]])
        scaled_input = (raw_input - self.X_min) / (self.X_max - self.X_min)

        # Step B: Convert to a PyTorch tensor
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32)

        # Step C: Ask the network (with no_grad because we aren't training)
        with torch.no_grad():
            prediction_eV = self.model(tensor_input).item()

        return prediction_eV

    def plot_surface(self, resolution=100):
        """Generates the diagnostic contour plot of the AI's learned physics."""
        print("[*] Generating AI Brain Diagnostic Plot...")

        T_test = np.linspace(100, 1000, resolution)
        Nd_test_log = np.linspace(10, 24, resolution)
        T_grid, Nd_grid = np.meshgrid(T_test, Nd_test_log)

        X_test_raw = np.column_stack((T_grid.flatten(), Nd_grid.flatten()))
        X_test_scaled = (X_test_raw - self.X_min) / (self.X_max - self.X_min)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        with torch.no_grad():
            Ef_pred_flat = self.model(X_test_tensor).numpy()

        Ef_pred_grid = Ef_pred_flat.reshape(resolution, resolution)

        plt.figure(figsize=(10, 7))
        contour = plt.contourf(T_grid, Nd_grid, Ef_pred_grid, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Predicted Fermi Level (eV)')
        plt.title('AI Surrogate Model: Diagnostic Surface')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Doping Level (Log10 $m^{-3}$)')
        plt.savefig("ai_brain_diagnostic.png", dpi=300, bbox_inches='tight')
        print("[*] Diagnostic plot saved to 'ai_brain_diagnostic.png'")
# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Boot the engine once
    engine = SiliconPhysicsEngine()

    # Query it as many times as you want
    test_T = 300
    test_Nd = 1e22

    ef_predicted = engine.predict_fermi_level(test_T, test_Nd)

    print("\n=== INFERENCE TEST ===")
    print(f"Temperature : {test_T} K")
    print(f"Doping (N_D): {test_Nd:.1e} m^-3")
    print(f"-> Predicted Fermi Level: {ef_predicted:.4f} eV")

    engine.plot_surface()