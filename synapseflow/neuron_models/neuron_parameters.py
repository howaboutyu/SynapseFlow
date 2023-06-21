from dataclasses import dataclass
from typing import List, Optional

from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuronParameters:
    """
    Defines the parameters for a neuron model.

    :param C_m: Membrane capacitance
    :type C_m: float
    :param E_L: Leak reversal potential
    :type E_L: float
    :param E_K: Potassium reversal potential
    :type E_K: float
    :param R_m: Membrane resistance
    :type R_m: float
    :param V_th: Threshold potential
    :type V_th: float
    :param V_reset: Reset potential
    :type V_reset: float
    :param V_peak: Peak potential, optional
    :type V_peak: Optional[float]

    :ivar tau_vc: Time constant for voltage clamp, optional
    :vartype tau_vc: Optional[float]
    :ivar tau_vth: Time constant for voltage increase method, optional
    :vartype tau_vth: Optional[float]
    :ivar V_th_max: Maximum voltage for V inc method, optional
    :vartype V_th_max: Optional[float]
    :ivar tau_G_ref: Time constant for conductance increase method, optional
    :vartype tau_G_ref: Optional[float]

    :ivar G_SRA: Spike rate adaptation conductance, optional
    :vartype G_SRA: Optional[float]
    :ivar Delta_G_SRA: Spike rate adaptation conductance delta, optional
    :vartype Delta_G_SRA: Optional[float]
    :ivar tau_SRA: Spike rate adaptation time constant, optional
    :vartype tau_SRA: Optional[float]
    :ivar V_max: Maximum voltage for spike adaptation, optional
    :vartype V_max: Optional[float]
    :ivar Delta_th: Spike adaptation threshold delta, optional
    :vartype Delta_th: Optional[float]
    :ivar a: Spike adaptation parameter, optional
    :vartype a: Optional[float]
    :ivar b: Spike adaptation parameter, optional
    :vartype b: Optional[float]
    """

    C_m: float  # membrane capacitance
    E_L: float  # leak reversal potential
    E_K: float  # potassium reversal potential
    R_m: float  # membrane resistance
    V_th: float  # threshold potential
    V_reset: float  # reset potential
    V_peak: Optional[float] = None  # peak potential

    # refractory period parameters
    tau_vc: Optional[float] = None  # time constant for voltage clamp
    tau_vth: Optional[float] = None  # time constant for voltage increase method
    V_th_max: Optional[float] = None  # maximum voltage for V inc method
    tau_G_ref: Optional[float] = None  # time constant for conductance increase method

    # spike adaptation parameters
    G_SRA: Optional[float] = None  # spike rate adaptation conductance
    Delta_G_SRA: Optional[float] = None  # spike rate adaptation conductance delta
    tau_SRA: Optional[float] = None  # spike rate adaptation time constant
    V_max: Optional[float] = None  # maximum voltage for spike adaptation
    Delta_th: Optional[float] = None  # spike adaptation threshold delta
    a: Optional[float] = None  # spike adaptation parameter
    b: Optional[float] = None  # spike adaptation parameter

    def __post_init__(self):
        """
        Initialize calculated parameters.

        :ivar G_L: Leak conductance
        :vartype G_L: float
        :ivar tau_m: Membrane time constant
        :vartype tau_m: float
        :ivar I_th: Threshold current
        :vartype I_th: float
        """
        self.G_L = 1.0 / self.R_m  # leak conductance
        self.tau_m = self.C_m / self.G_L  # membrane time constant
        self.I_th = self.G_L * (self.V_th - self.E_L)  # threshold current
