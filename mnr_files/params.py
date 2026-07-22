import numpy as np

# Physical constants

hbar = 6.582119569e-16       # Reduced Planck constant in eV*s
k_B = 8.617333262145e-5      # Boltzmann constant in eV/K
g = 19e-6                    # QD-cavity coupling in eV (19 μeV)
kappa = 90e-6                # Cavity linewidth in eV (90 μeV)
gamma_0 = hbar / 1e-9        # Bulk decay rate in eV (~6.685185 μeV)
delta = 0                    # Detuning between QD and cavity (assuming zero for resonance)
alpha = 0.1e-6               # Pure dephasing coefficient in eV (0.1 μeV)
epsilon_p = 1e-3             # Energy of maximally coupled phonons in eV (1 meV)

# Source parameters

R_SPS=10^9                   # Source repetition rate, in Hz

# Attocube cryostat characteristics, used as default cryostat parameters

T_cryo=2.8                   # Cryostat temperature
cooling_power=0.05           # cooling power, in Watts
cryo_efficiency=0.0038       # coefficient to Carnot efficiency
det_per_cryo=26              # Number of detectors per cryostat

# Other power sources

extra_power_consumption=1500 # in Watts

# Transmission efficiencies

bof = 0.85                   # brightness output fiber: includes first length brightness + filters + single mode coupling
eta_snspd=0.99               # SNSPD efficiency 
eta_dmx=0.99                  # Demultiplexer efficiency  
cfwg=0.3                     # Attenuation at fiber - waveguide junction (in dB)
eta_fwg = 10**(-2*cfwg/10)   # Efficiency of insertion 
cmzi=0.05                   # Attenuation per MZI (in dB)

# Error correction

t_ecc=10**(-5)              # Time of an error correction cycle

