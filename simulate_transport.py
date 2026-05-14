import numpy as np
import physics_tools as pt
from materials import Silicon, k_B, q, GalliumArsenide, SiGe
from inference import SiliconPhysicsEngine
material = SiGe(0.375)
def main():
    print("=== TRANSPORT & CONDUCTIVITY SIMULATOR ===")

    engine = SiliconPhysicsEngine()

    Temp = 300.0
    N_D = 1e22

    print(f"\nConditions: {Temp}K, N_D = {N_D:.1e} m^-3")

    Ef_eV = engine.predict_fermi_level(Temp, N_D)
    print(f"AI Predicted E_f: {Ef_eV:.4f} eV")

    Nc = Silicon.get_Nc(Temp)
    Nv = Silicon.get_Nv(Temp)
    Ec_eV = Silicon.get_bandgap_j(Temp) / q
    Ev_eV = 0.0
    k_B_eV = k_B / q

    n = Nc * np.exp(-(Ec_eV - Ef_eV) / (k_B_eV * Temp))
    p = Nv * np.exp(-(Ef_eV - Ev_eV) / (k_B_eV * Temp))


    N_D_cm3 = N_D * 1e-6
    N_A_cm3 = 0.0
    N_total_cm3 = N_D_cm3 + N_A_cm3


    mu_e = pt.calc_arora_mobility(Temp, N_total_cm3, Silicon.arora_e)
    mu_h = pt.calc_arora_mobility(Temp, N_total_cm3, Silicon.arora_h)

    sigma = q * (n * mu_e + p * mu_h)

    rho = 1.0 / sigma

    print("\n--- Transport Results ---")
    print(f"Electrons (n): {n:.2e} m^-3")
    print(f"Holes (p)    : {p:.2e} m^-3")
    print(f"Mobility (ue): {mu_e:.4f} m^2/V*s")
    print(f"Conductivity : {sigma:.2f} S/m")
    print(f"Resistivity  : {rho:.4f} Ohm-m")


if __name__ == "__main__":
    main()