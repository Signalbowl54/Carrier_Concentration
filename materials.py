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

Germanium = Semiconductor(
    name='Germanium',
    Eg_0_ev=0.742,
    alpha=4.77e-4,
    beta=235,
    me_eff_dos=0.36,
    mh_eff_dos=0.81,
    me_eff_cc=0.12,
    mh_eff_cc=0.34,
    arora_e={
        'mu_min': 700, 'mu_max': 3900, 'N_ref': 1.75e17, 'alpha_m': 0.68,
        'ex_min': -0.57, 'ex_max': -2.33, 'ex_N': 2.4, 'ex_a': -0.146
    },
    arora_h={
        'mu_min': 450.0, 'mu_max': 1900, 'N_ref': 8.25e17, 'alpha_m': 0.45,
        'ex_min': -0.57, 'ex_max': -2.33, 'ex_N': 2.4, 'ex_a': -0.146
    }
)
# --- Alloys ---
class SiGe(Semiconductor):
    def __init__(self, x):
        self.x = x

        self.alpha = (1-x)*4.73e-4 + x*4.77e-4
        self.beta = (1-x)*636 + x*235

        # Get Eg_0_ev for Ge-like (x>=.85) or Si-like (x<.85)
        if x < 0.85:
            self.Eg_0_ev = 1.17 - 0.43*x + 0.206*x**2
        else:
            self.Eg_0_ev = 0.945 - 0.204*x

        if x < 0.85:
            self.me_eff_dos = 1.06
        else:
            self.me_eff_dos = 1.55

        if x < 0.85:
            self.me_eff_cc = 0.26
        else:
            self.me_eff_cc = 0.12

        self.mh_eff_dos = (1-x) * 0.81 + x * 0.34
        self.mh_eff_cc = (1-x) * 0.37 + x * 0.21

    @property
    def arora_e(self):
            # Linear interpolation of the Silicon and Germanium endpoints
        return {
            key: (1 - self.x) * Silicon.arora_e[key] + self.x * Germanium.arora_e[key]
            for key in Silicon.arora_e
        }

    @property
    def arora_h(self):
        return {
            key: (1 - self.x) * Silicon.arora_h[key] + self.x * Germanium.arora_h[key]
            for key in Silicon.arora_h
        }












































