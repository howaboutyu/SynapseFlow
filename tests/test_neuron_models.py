from context import *
import numpy as np

from synapseflow.neuron_models.lif_model import LIFModel, AELIFModel
from synapseflow.neuron_models.neuron_parameters import NeuronParameters


def test_LIF():
    C_m = 2.0e-9  # membrane capacitance
    E_L = -70.0e-3  # leak reversal potential
    E_K = -80e-3  # potassium reversal potential
    R_m = 5.0e6  # membrane resistance
    G_L = 1.0 / R_m  # leak conductance
    V_th = -50.0e-3  # threshold potential
    V_reset = -65.0e-3  # reset potential

    params = NeuronParameters(
        C_m=C_m, E_L=E_L, E_K=E_K, R_m=R_m, V_th=V_th, V_reset=V_reset
    )

    dt = 0.1e-3
    times = np.arange(0, 1, dt)

    I = np.zeros_like(times)

    I[100:500] = params.I_th * 1.1

    lif = LIFModel(neuron_parameters=params, times=times, I=I)
    lif.simulate()

    assert lif.num_fire > 0


def test_AELIF():
    params = NeuronParameters(
        C_m=0.1e-9,
        E_L=-70.0e-3,
        R_m=100.0e6,
        E_K=-80.0e-3,
        V_th=-50e-3,
        V_reset=-65.0e-3,
        V_peak=50.0e-3,
        V_th_max=200.0e-3,
        tau_SRA=150.0e-3,
        a=2.0e-9,
        b=0.0,
        Delta_th=2.0e-3,
        V_max=200.0e-3,
    )
    
    dt = 0.1e-3
    times = np.arange(0, 1, dt)
    I = np.zeros_like(times)
    
    aelif_no_fire = AELIFModel(neuron_parameters=params, times=times, I=I)
    aelif_no_fire.simulate()

    aelif_fire = AELIFModel(neuron_parameters=params, times=times, I=I + params.I_th * 1.1)
    aelif_fire.simulate()

    assert aelif_no_fire.num_fire == 0
    assert aelif_fire.num_fire > 0
    
