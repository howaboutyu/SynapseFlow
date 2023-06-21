from typing import Dict, List, Union
import numpy as np
from dataclasses import dataclass, field

from .neuron_parameters import NeuronParameters
from .base_model import BaseNeuronModel


@dataclass
class LIFModel(BaseNeuronModel):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    :param neuron_parameters: Neuron model parameters
    :type neuron_parameters: NeuronParameters
    :param times: Simulation time array
    :type times: np.ndarray
    :param I: Input current array
    :type I: np.ndarray
    :param noise_sigma: Standard deviation of noise, defaults to 0.0
    :type noise_sigma: float, optional
    :ivar noises: Noise array, automatically initialized after object creation
    :vartype noises: np.ndarray
    """

    # Euler-Mayamara noise
    noise_sigma: float = 0.0
    noises: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Initialize the model, creating the noise array and setting initial values.
        """
        super().__post_init__()
        self.noises = (
            np.random.normal(size=self.times.shape)
            * self.noise_sigma
            * np.sqrt(self.dt)
        )

    def dvdt(self, V_m, I_app) -> float:
        """
        Calculate the derivative of voltage with respect to time.

        :param V_m: Membrane potential at the current time
        :type V_m: float
        :param I_app: Applied current at the current time
        :type I_app: float
        :return: Derivative of voltage with respect to time
        :rtype: float

        .. math::
            \\frac{dV}{dt} = \\frac{(E_L - V_m) / R_m + I_{app}}{C_m}
        """
        _dvdt = (self.neuron_parameters.E_L - V_m) / self.neuron_parameters.R_m + I_app
        _dvdt = _dvdt / self.neuron_parameters.C_m
        return _dvdt

    def simulate(self):
        """
        Simulate the neuron model.

        Uses the Euler method to solve the differential equation, with a noise term added.
        If the membrane potential exceeds the threshold, it is reset and a spike is counted.
        """
        for i in range(1, len(self.times)):
            V_new = self.V[i - 1] + self.dvdt(self.V[i - 1], self.I[i - 1]) * self.dt

            if isinstance(self.noises, np.ndarray):
                V_new = V_new + self.noises[i]

            if V_new > self.neuron_parameters.V_th:
                V_new = self.neuron_parameters.V_reset
                self.num_fire += 1
                print(f"Neuron fired at time {self.times[i]}")
                self.last_fire_time = self.times[i]

            self.V[i] = V_new


@dataclass
class AELIFModel(LIFModel):
    """
    Adaptive Exponential LIF (AELIF) neuron model.

    Inherits from LIFModel, adds spike rate adaptation.

    :param neuron_parameters: Neuron model parameters
    :type neuron_parameters: NeuronParameters
    :param times: Simulation time array
    :type times: np.ndarray
    :param I: Input current array
    :type I: np.ndarray
    :param noise_sigma: Standard deviation of noise, defaults to 0.0
    :type noise_sigma: float, optional
    :ivar I_SRA_array: Spike rate adaptation current array, automatically initialized after object creation
    :vartype I_SRA_array: np.ndarray
    :ivar last_spike_time: Time of the last spike, automatically initialized after object creation
    :vartype last_spike_time: float
    :ivar isi_array: Inter-spike interval array
    :vartype isi_array: List[float]
    """


    I_SRA_array: np.ndarray = field(init=False)  # spike rate adaptation current
    last_spike_time: float = field(init=False)  # last spike time
    isi_array: List[float] = field(default_factory=list)  # inter-spike interval
    spike_array: np.ndarray = field(init=False)  # spike array, 1 if spike, 0 if no spike, same length as times

    def __post_init__(self):
        """
        Initialize the model, creating the I_SRA_array and setting initial values.
        """
        super().__post_init__()
        self.I_SRA_array = np.zeros_like(self.times)
        self.last_spike_time = -np.inf
        self.spike_array = np.zeros_like(self.times)
        
        
    def dI_SRA_dt(self, V_m, I_SRA):
        """
        Calculate the change in spike rate adaptation current with respect to time.

        :param V_m: Membrane potential at the current time
        :type V_m: float
        :param I_SRA: Spike rate adaptation current at the current time
        :type I_SRA: float
        :return: Change in spike rate adaptation current with respect to time
        :rtype: float

        .. math::
            \\frac{dI_{SRA}}{dt} = \\frac{a * (V_m - E_L) - I_{SRA}}{\\tau_{SRA}}
        """
        a = self.neuron_parameters.a
        _dI_SRA_dt = a * (V_m - self.neuron_parameters.E_L) - I_SRA

        return _dI_SRA_dt / self.neuron_parameters.tau_SRA

    def dvdt(self, V_m, I_app, I_SRA):
        """
        Calculate the change in membrane potential with respect to time.

        :param V_m: Membrane potential at the current time
        :type V_m: float
        :param I_app: Applied current at the current time
        :type I_app: float
        :param I_SRA: Spike rate adaptation current at the current time
        :type I_SRA: float
        :return: Change in membrane potential with respect to time
        :rtype: float

        .. math::
            \\frac{dV}{dt} = \\frac{G_L * (E_L - V_m + \\Delta_{th} * e^{\\frac{(V_m - V_{th})}{\\Delta_{th}}}) - I_{SRA} + I_{app}}{C_m}
        """
        exp_term = self.neuron_parameters.Delta_th * np.exp(
            (V_m - self.neuron_parameters.V_th) / self.neuron_parameters.Delta_th
        )

        _dv_dt = (
            self.neuron_parameters.G_L * (self.neuron_parameters.E_L - V_m + exp_term)
            - I_SRA
            + I_app
        )

        return _dv_dt / self.neuron_parameters.C_m

    def simulate(self):
        """
        Simulate the neuron model.

        Uses the Euler method to solve the differential equation, including a spike rate adaptation current. If the membrane potential exceeds the maximum, it is reset, a spike is counted, and the spike rate adaptation current is increased.

        The AELIF model uses the rule :math:`V_m > V_{max}` then :math:`V_m=V_{reset}` and :math:`I_{IRA}=I_{IRA}+b`
        
        """

        for i in range(1, len(self.times)):
            self.V[i] = self.V[i - 1] + self.dt * self.dvdt(
                self.V[i - 1], self.I[i - 1], self.I_SRA_array[i - 1]
            )

            self.I_SRA_array[i] = self.I_SRA_array[i - 1] + self.dt * self.dI_SRA_dt(
                self.V[i - 1], self.I_SRA_array[i - 1]
            )

            if self.V[i] >= self.neuron_parameters.V_max:
                self.V[i] = self.neuron_parameters.V_reset
                self.I_SRA_array[i] += self.neuron_parameters.b
                self.num_fire += 1

                self.isi_array.append(self.times[i] - self.last_spike_time)

                self.last_spike_time = self.times[i]
                self.spike_array[i] = 1


