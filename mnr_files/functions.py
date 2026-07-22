import numpy as np
from mnr_files.params import *

# Quantum dot functions

def n_th(T=T_cryo):
    '''
    Thermal (Bose-Einstein) occupation number of the maximally coupled phonon mode at
    temperature T.
    '''
    theta = epsilon_p / (k_B * T)
    return 1.0 / (np.exp(theta) - 1)


def gamma_star(T=T_cryo):
    n_ep = n_th(T)
    gamma_star_T_val = alpha * n_ep * (n_ep + 1)
    return gamma_star_T_val


def F_eff(T=T_cryo):
    '''
    Calculates the effective Purcell factor F_eff for a given temperature T. According to the supplemental material of the paper "Reducing Phonon induced decoherence in solid-state single-photon sources with cavity quantum electrodynamics" by T. Grange et al. ().

    Parameters:
        T (float): Temperature.
    '''
    gamma_0_F_eff = 2 * g**2 / (kappa + gamma_0 + gamma_star(T))
    F_eff_val = gamma_0_F_eff / gamma_0
    return F_eff_val

def eta_ZPL(T=T_cryo):
    return (0.76 - 0.99) / 20 * T + 0.99

def gamma_T(T=T_cryo):
    gamma_T_val = gamma_0 + eta_ZPL(T) * F_eff(T) * gamma_0
    return gamma_T_val

def I_ZPL(T=T_cryo):
    I_ZPL_val = gamma_T(T) / (gamma_T(T) + gamma_star(T))
    return np.maximum(I_ZPL_val, 0) 

def I(T=T_cryo):
    return I_ZPL(T) - 0.05

# Cryogenics and QPU functions: all photonics

def P_cryo(nd,T=T_cryo,P0=cooling_power,x=cryo_efficiency,n_det_max=det_per_cryo):
    '''
    Calculates the power of the cryostat for a given number of detectors nd.
    We assume here that we can store 12 detectors in the cryostat.

    Parameters:
        nd (int): Number of detectors in cryostat.
        T (float): Cryostat temperature.
        P0 (flot): cooling power
        x (float): coefficient to the Carnot efficiency; the total efficiency is x * Carnot efficiency 
        n_det_max (int): maximum number of detectors per cryostat
        

    Returns:
        float: Power consumption of the cryostat, in Watts.
    '''
    n_c=(1+nd/n_det_max) //1 # number of cryostats
    
    return (300 - T) / (x*T) * P0*n_c

def P_tot(nd,T=T_cryo,P0=cooling_power,x=cryo_efficiency,n_det_max=det_per_cryo,y=extra_power_consumption):
    '''
    Calculates the total power consumption 

    Parameters:
        nd (int): Number of detectors
        T (float): Cryostat temperature.
        P0 (flot): cooling power
        x (float): coefficient to the Carnot efficiency; the total efficiency is x * Carnot efficiency 
        n_det_max (int): maximum number of detectors per cryostat
        y (float): extra power consumption (laser + circuitry + classical CPU)
        

    Returns:
        float: Total power consumption, in Watts.
    '''
    n_c=(1+nd/det_per_cryo) //1 # number of cryostats
    
    return P_cryo(nd,T,P0,x,n_det_max) + y 

def P_cryo_v2(nd,ns,T=T_cryo,x=cryo_efficiency):
    '''
    Calculates the power of the cryostat for a given number of detectors nd.
    We assume here that we can store 12 detectors in the cryostat.

    Parameters:
        nd (int): Number of detectors in cryostat.
        nd (int): Number of source in cryostat.
        T (float): Cryostat temperature.
        x (float): coefficient to the Carnot efficiency; the total efficiency is x * Carnot efficiency 
        

    Returns:
        float: Power consumption of the cryostat, in Watts.
    '''
    P0=50*10**(-6)*nd+10*10**(-3)*ns # Cooling power in W
    
    return (300 - T) / (x*T) * P0

def P_tot_v2(nd,ns,nc,nm,T=T_cryo,x=cryo_efficiency):
    '''
    Calculates the total power consumption 

    Parameters:
        nd (int): Number of detectors
        ns (int): number of sources
        nc (int): number of phase shifters
        nm (int): number of measurements
        T (float): Cryostat temperature.
        x (float): coefficient to the Carnot efficiency; the total efficiency is x * Carnot efficiency 
        
    Returns:
        float: Total power consumption, in Watts.
    '''
    P_laser=5*ns
    P_mem=(nc+nm)*32 
    P_CPU=250 
    P_DMX=100*ns
    P_extra=100
    
    return P_cryo_v2(nd,ns,T,x) + P_laser + P_mem + P_CPU + P_DMX + P_extra 

def transmission(nd,b_of=bof,snspd=eta_snspd,dmx=eta_dmx,c_mzi=cmzi,fiber_wg=eta_fwg):
    '''
    Calculates the end to end transmission, assuming that the number of components seen by 1 photon is equal to the number of detectors (good    estimate for the current architecture)

    Parameters:
        bof (float): brightness outside fiber
        snspd (float): efficiency of SNSPDs
        dmx (float): efficiency of DMX
        nd (int): number of detectors / ciruit depth
        c_mzi (float): loss factor in dB per MZI 
        fiber_wg (float): loss at the fiber to waveguide junction
    '''
    return bof*snspd*dmx*fiber_wg*10**(-c_mzi*nd/10)

def transmitted_photons(c_mzi,c_fiber_wg,nph,nd):
    '''
    Calculates the number of detected photons in the output of the boson sampler.

    Parameters:
        nph (int): number of input photons.
        nd (int): number of detectors
        c_mzi (float): loss factor in dB per MZI 
        c_fiber_wg (float): loss factor at the fiber to waveguide junction

    Returns:
        float: Number of detected photons.
    '''
   
    return nph*transmission(c_mzi,c_fiber_wg,nd)


def coincidence_rate(nph,eta,r=R_SPS): 
    '''
    input:
    r_sps (float): Repetition Rate of the Laser
    nph (int): number of photons
    c_mzi (float): loss factor in dB per MZI 
    c_fiber_wg (float): loss factor at the fiber to waveguide junction
    nd (int): number of detectors/optical components on the chip
 
    output:
    Coincidence Rate in Hz


    '''

    return r / nph *eta ** nph 

def time_per_post_sel_sample(nph,nd,eta,nps,r=R_SPS, n_det_max=det_per_cryo):
    '''
    input:
    r (float): Repetition Rate of the Laser
    eta (float): end to end transmission
    nph (int): number of photons
    nd (int): number of detectors
    nps (int:: number of photons to postselect on

    output:
    energy required to produce a sample of nph detected photons, in Wh
    '''
    n_s=(1+nd/n_det_max) //1 # number of sources
    n_eff=np.ceil(nph/n_s)
    return n_eff/(r*eta**nps)

def time_per_post_sel_sample_v2(nph,nd,ns,eta,nps,r=R_SPS):
    '''
    input:
    r (float): Repetition Rate of the Laser
    eta (float): end to end transmission
    nph (int): number of photons
    nd (int): number of detectors
    ns (int): number of sources
    nps (int:: number of photons to postselect on

    output:
    energy required to produce a sample of nph detected photons, in Wh
    '''
    n_eff=np.ceil(nph/ns)
    return n_eff/(r*eta**nps)
   

def energy_per_post_sel_sample(nph,nd,eta,nps,T=T_cryo,r=R_SPS,P0=cooling_power,x=cryo_efficiency,n_det_max=det_per_cryo,y=extra_power_consumption):
    '''
    input:
    T (float): cryostat temperature
    r_sps (float): Repetition Rate of the Laser
    eta (float): end to end transmission
    nph (int): number of photons
    nd (int): number of detectors
    nps (int:: number of photons to postselect on

    output:
    energy required to produce a sample of nph detected photons, in Wh
    '''
    n_s=(1+nd/n_det_max) //1 # number of sources
    n_eff=np.ceil(nph/n_s)
    return n_eff/(r*eta**nps) * P_tot(nd,T,P0,x,n_det_max,y) / (3.6*10**3)
    
def energy_per_sample(nph,nd,T=T_cryo,r=R_SPS,P0=cooling_power,x=cryo_efficiency,n_det_max=det_per_cryo,y=extra_power_consumption):
    '''
    input:
    T (float): cryostat temperature
    r_sps (float): Repetition Rate of the Laser
    eta (float): end to end transmission
    nph (int): number of photons
    nd (int): number of detectors

    output:
    energy required to produce a sample of nph photons, without post selection, in Wh
    '''
    n_s=(1+nd/n_det_max) //1 # number of sources
    n_eff=np.ceil(nph/n_s)
    return n_eff/(r) * P_tot(nd,T,P0,x,n_det_max,y) / (3.6*10**3)

# Cryogenics and QPU functions: SPOQC

def P_cryo_SPOQC(ns,nd,nrus,T=T_cryo,x=cryo_efficiency):
    '''
    Calculates the power of the cryostat for a given number of detectors nd.
    We assume here that we can store 12 detectors in the cryostat.

    Parameters:
        nd (int): Number of detectors in cryostat.
        nd (int): Number of source in cryostat.
        nrus (int): Number of RUS gates per source
        T (float): Cryostat temperature.
        x (float): coefficient to the Carnot efficiency; the total efficiency is x * Carnot efficiency 
        

    Returns:
        float: Power consumption of the cryostat, in Watts.
    '''
    P0=50*10**(-6)*10*nd+10*10**(-3)*ns+10**(-4)*nrus*ns # Cooling power in W
    
    return (300 - T) / (x*T) * P0

def P_tot_SPOQC(ns,nd,nrus,nps,T=T_cryo,x=cryo_efficiency):
    '''
    Calculates the total power consumption 

    Parameters:
        nd (int): Number of detectors
        ns (int): number of sources
        nrus (int): Number of RUS gates per source
        nc (int): number of phase shifters
        T (float): Cryostat temperature.
        x (float): coefficient to the Carnot efficiency; the total efficiency is x * Carnot efficiency 
        
    Returns:
        float: Total power consumption, in Watts.
    '''
    P_laser=5*ns
    P_mem=(nps+nd)*32 
    P_CPU=250 
    P_extra=100
    
    return P_cryo_SPOQC(ns,nd, nrus,T,x) + P_laser + P_mem + P_CPU + P_extra 


# Rotated surface code functions

def num_sources(d):
    '''
    Calculates the number of sources required to implement a rotated surface code of distance d
    '''
    return d**2

def num_det(d):
    '''
    Calculates the number of PNRs required to implement a rotated surface code of distance d
    '''
    ns=num_sources(d)
    return 6*ns

def num_rus(d):
    '''
    Calculates the number of RUS gates required to implement a rotated surface code of distance d
    '''
    ns=num_sources(d)
    return ns

def num_ps(d):
    '''
    Calculates the number of phase shifters required to implement a rotated surface code of distance d
    '''
    ns=num_sources(d)
    nrus=num_rus(d)
    return ns+nrus

def params_SPOQC(d):
    ns=num_sources(d)
    nd=num_det(d)
    nrus=num_rus(d)
    nps=num_ps(d)
    return ns,nd,nrus,nps

def energy_ecc(d,T=T_cryo,x=cryo_efficiency,t=t_ecc):
    ns,nd,nrus,nps=params_SPOQC(d)
    P=P_tot_SPOQC(ns,nd,nrus,nps,T,x)
    return t_ecc*P






####### A DEFINIR   ############

import sys, os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "program"))

from QEC import get_IXYZ_error_proba, fit_threshold_mc

_THRESHOLD_FIT = None

def threshold_fit(): # compute & save
    global _THRESHOLD_FIT
    if _THRESHOLD_FIT is None:
        A, B, p_th, _ = fit_threshold_mc(verbose=False, root=_REPO_ROOT)
        _THRESHOLD_FIT = (A, B, p_th)
    return _THRESHOLD_FIT


def logical_error(d, T=T_cryo, t=t_ecc):
    A, B, p_th = threshold_fit()

    g_rate  = (gamma_0 * n_th(T) + gamma_star(T)) / hbar 
    gc_rate = (2 * gamma_0 * n_th(T)) / hbar             
    
    _, pX, pY, pZ = get_IXYZ_error_proba.pyfunc(t, g_rate, gc_rate)
    p = pX + pY + pZ

    return np.minimum(A * (p / p_th) ** (B * (d + 1) / 2), 0.75)

########################






    
     
# Classical cost of boson sampling

def FLOP_of_permament(N):
    return 2**N * N

def FLOP_lowerbound_CliffordClifford(m, n): # Assumption: x^j and second_perm_term and normalization factor (n over m)
    num_FLOP = 0
    for j in range(2, n + 1):
        num_FLOP += FLOP_of_permament(j) + m * j
    return num_FLOP

def FLOP_lowerbound(m, n, method="CliffordClifford"):
    method_to_function = {
        "CliffordClifford": FLOP_lowerbound_CliffordClifford,
    }

    FLOP_function = method_to_function.get(method)

    if FLOP_function is None:
        raise ValueError(f"Unknown method: {method}")

    return max(1,FLOP_function(m, n))

def classical_energy_per_sample(m, n, eff, method):
    return FLOP_lowerbound(m, n, method) / eff 

def classical_powerconsumption(m, n, eff, R_max, method):
    return classical_energy_per_sample(m, n, eff, method) / (FLOP_lowerbound(m, n, method) / R_max) 

        