"""
Utilities useful for CNS
"""

import numpy as np

def expand_bins(spike_array: np.ndarray, new_dt: float, old_dt: float) -> np.ndarray:
    """
    Reduces the number of bins

    :params spike_array: The spike array, an array with zeros, where 1 indicates a spike
    :type spike_array: np.ndarray
    :params new_dt: The new time step
    :type new_dt: float
    :params old_dt: The old time step
    :type old_dt: float
    :return: The expanded spike array
    :rtype: np.ndarray

    """

    sf = int(new_dt/old_dt)
    spike_array = np.split(spike_array, sf, axis=0)
    spike_array = np.array(spike_array)
    spike_array = np.mean(spike_array, axis=0)
    return spike_array


def STA(
    currents: np.ndarray,
    spike_array: np.ndarray,
    dt: float,
    t_minus: float,
    t_plus: float
) -> np.ndarray:
    """

    Spike triggered average

    :params currents: The applied currents
    :type currents: np.ndarray
    :params spike_array: The spike array, an array with zeros, where 1 indicates a spike
    :type spike_array: np.ndarray
    :params dt: The time step
    :type dt: float
    :params t_minus: The time before the spike
    :type t_minus: float
    :params t_plus: The time after the spike
    :type t_plus: float
    :return: The STA
    :rtype: np.ndarray

    """

    t_minus = int(t_minus/dt)
    t_plus = int(t_plus/dt)

    sta = np.zeros((t_minus + t_plus, 1))

    for i in range(len(spike_array)):
        if spike_array[i] == 1:
            if i - t_minus < 0:
                sta[0:i] += currents[0:i]
            else:
                sta += currents[i-t_minus:i+t_plus]


    sta = sta / np.sum(spike_array)

    return sta



def find_fano_factor(time_range, spike_locations, dt):
    """
    Calculates the Fano factor based on spike locations within time windows.

    The Fano factor is defined as :math:`\\frac{\\sigma^2}{\\mu}`, where :math:`\\sigma^2` is the variance and :math:`\\mu` is the mean.

    :param time_range: The duration of each time window in milliseconds.
    :type time_range: int or float
    :param spike_locations: Array of spike locations (0 or 1) indicating spike occurrences.
    :type spike_locations: np.ndarray
    :param dt: The time step size in milliseconds.
    :type dt: int or float
    :return: The Fano factor.
    :rtype: float or None
    """

    num_steps_per_window = int(time_range / dt)

    # Check if we can split the spike_locations into equal windows
    if len(spike_locations) % num_steps_per_window != 0:
        return None

    # Split spike_locations into windows
    spike_windows = np.split(spike_locations, num_steps_per_window, axis=0)
    spike_windows = np.asarray(spike_windows)

    # Count the number of spikes in each window
    spike_counts = np.sum(spike_windows, axis=1)

    # Calculate the Fano factor
    fano_factor = np.var(spike_counts) / np.mean(spike_counts)

    return fano_factor
