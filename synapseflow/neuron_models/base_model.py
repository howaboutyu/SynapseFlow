from dataclasses import dataclass, field
from typing import Dict, List, Union
import numpy as np

from .neuron_parameters import NeuronParameters


@dataclass
class BaseNeuronModel:
    """
    Base class for all neuron models.

    :param neuron_parameters: Parameters for neuron model
    :type neuron_parameters: NeuronParameters
    :param times: Array of time steps for simulation
    :type times: np.ndarray
    :param I: Input current to neuron
    :type I: np.ndarray
    """

    neuron_parameters: NeuronParameters
    times: np.ndarray

    I: np.ndarray
    V: np.ndarray = field(init=False)
    dt: float = field(init=False)
    num_fire: int = field(init=False, default=0)

    def __post_init__(self):
        """
        Set default values for parameters.
        Initializes voltage array and timestep size.
        """

        self.V = np.zeros_like(self.times)
        self.V[0] = self.neuron_parameters.E_L
        self.dt = self.times[1] - self.times[0]

    @property
    def fire_rate(self):
        """
        Return the firing rate of the neuron.

        :return: Firing rate of the neuron
        :rtype: float
        """

        return self.num_fire / (self.times[-1] - self.times[0])

    def dVdt(self, step: int) -> float:
        """
        Calculate the change in voltage at a given time step.

        :param step: Index of time step
        :type step: int
        :return: Change in voltage at given time step
        :rtype: float
        """

        raise NotImplementedError

    def simulate(self):
        """
        Simulate the neuron model.
        Needs to be implemented in subclasses.
        """

        raise NotImplementedError
