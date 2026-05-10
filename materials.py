import numpy as np
from dataclasses import dataclass

# --- Universal Constants ---
q = 1.602176634e-19       # Electron charge (C)
k_B = 1.380649e-23        # Boltzmann constant (J/K)
h = 6.62607015e-34        # Planck constant (J*s)
m_0 = 9.1093837015e-31    # Rest mass of electron (kg)

@dataclass
class Semiconductor:
    """
    A 'Smart' Data Object that calculates temperature-dependent
    properties on the fly using empirical models.
    """
    name: str
    Eg_0_ev: float          # Bandgap at 0K in eV
    alpha: float            # Varshni parameter (eV/K)
    beta: float             # Varshni parameter (K)
    me_eff_ratio: float     # Effective electron mass (relative to m_0)
    mh_eff_ratio: float     # Effective hole mass (relative to m_0)

    def get_bandgap_j(self, T: float) -> float:
        """
        Calculates the Bandgap in J using Varshni's equation
        :param T: Temperature in Kelvin
        :return: Bandgap energy in J
        """
        Eg_T_ev = self.Eg_0_ev - (self.alpha * T**2) / (T + self.beta)
        return Eg_T_ev * q

    def get_Nc(self, T: float) -> float:
        """
        Calculates the Effective Density of States in Conduction Band (m^-3)
        :param T: Temperature in K
        :return: Effective Density of States in Conduction Band (m^-3)
        """
        m_e = self.me_eff_ratio * m_0
        Nc = 2.0 * ((2.0 * np.pi * m_e * k_B * T)/ (h**2))**(3/2)
        return Nc

    def get_Nv(self, T: float) -> float:
        """
        Calculates the Effective Density of States in Valence Band (m^-3)
        :param T: Temperature in K
        :return: Effective Density of States in Valence Band (m^-3)
        """
        m_e = self.mh_eff_ratio * m_0
        Nv = 2.0 * ((2.0 * np.pi * m_e * k_B * T) / (h**2))**(3/2)
        return Nv


# --- Materials Table ---

Silicon = Semiconductor(
    name='Silicon',
    Eg_0_ev=1.170,
    alpha=4.73e-4,
    beta=636,
    me_eff_ratio=1.18,
    mh_eff_ratio=0.81,
)

GalliumArsenide = Semiconductor(
    name='Gallium Arsenide',
    Eg_0_ev=1.519,
    alpha=5.41e-4,
    beta=204,
    me_eff_ratio=0.067,
    mh_eff_ratio=0.53
)

















































