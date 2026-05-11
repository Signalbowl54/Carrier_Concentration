import numpy as np
from dataclasses import dataclass, field

# --- Universal Constants ---
q = 1.602176634e-19       # Electron charge (C)
k_B = 1.380649e-23        # Boltzmann constant (J/K)
h = 6.62607015e-34        # Planck constant (J*s)
h_bar = 1.0545717e-34      # Reduced Planck Constant
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
    me_eff_dos: float       # Effective electron mass (relative to m_0)
    mh_eff_dos: float       # Effective hole mass (relative to m_0)
    me_eff_cc: float        # Conductivity Effective Mass (relative to m_0)
    mh_eff_cc: float


    arora_e: dict = field(default_factory=dict)
    arora_h: dict = field(default_factory=dict)


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
        m_e = self.me_eff_dos * m_0
        Nc = 2.0 * ((2.0 * np.pi * m_e * k_B * T)/ (h**2))**(3/2)
        return Nc

    def get_Nv(self, T: float) -> float:
        """
        Calculates the Effective Density of States in Valence Band (m^-3)
        :param T: Temperature in K
        :return: Effective Density of States in Valence Band (m^-3)
        """
        m_e = self.mh_eff_dos * m_0
        Nv = 2.0 * ((2.0 * np.pi * m_e * k_B * T) / (h**2))**(3/2)
        return Nv


# --- Materials Table ---

Silicon = Semiconductor(
    name='Silicon',
    Eg_0_ev=1.170,
    alpha=4.73e-4,
    beta=636,
    me_eff_dos=1.18,
    mh_eff_dos=0.81,
    me_eff_cc=0.26,
    mh_eff_cc=0.37,
    arora_e = {
        'mu_min': 88.0, 'mu_max': 1252.0, 'N_ref': 1.26e17, 'alpha_m': 0.88,
        'ex_min': -0.57, 'ex_max': -2.33, 'ex_N': 2.4, 'ex_a': -0.146
    },
    arora_h = {
        'mu_min': 54.3, 'mu_max': 407.0, 'N_ref': 2.35e17, 'alpha_m': 0.88,
        'ex_min': -0.57, 'ex_max': -2.23, 'ex_N': 2.4, 'ex_a': -0.146
    }
)

GalliumArsenide = Semiconductor(
    name='Gallium Arsenide',
    Eg_0_ev=1.519,
    alpha=5.41e-4,
    beta=204,
    me_eff_dos=0.067,
    mh_eff_dos=0.53,
    me_eff_cc=0.063,
    mh_eff_cc=0.34,
    # TCAD Standard Caughey-Thomas Parameters for GaAs
    # Note: GaAs temperature dependence is driven mostly by polar optical phonons,
    # which usually gives an exponent near -1.0 for electrons.
    arora_e = {
        'mu_min': 0.0, 'mu_max': 8500.0, 'N_ref': 1.69e17, 'alpha_m': 0.436,
        'ex_min': 0.0, 'ex_max': -1.0, 'ex_N': 0.0, 'ex_a': 0.0
    },
    arora_h = {
        'mu_min': 44.0, 'mu_max': 400.0, 'N_ref': 2.75e17, 'alpha_m': 0.395,
        'ex_min': 0.0, 'ex_max': -2.1, 'ex_N': 0.0, 'ex_a': 0.0
    }
)

















































