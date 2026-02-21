from typing import List, Tuple

def linear_movement(num_neurons: int, simulation_time: float, speed: float) -> Tuple[List[int], List[float]]:
    """
    Generates input neuron spikes linearly distributed in time.
    :param num_neurons: Number of neurons (receptors) in the series.
    :param simulation_time: Total simulation time in [s].
    :param speed: Pulse movement speed in [neurons/s].
    :return: Lists of neuron spikes (ids) and the corresponding timestamps.
    """

    min_id = 0
    max_id = num_neurons - 1

    current_id = min_id
    step_dir = 1
    time_step = 1.0 / speed # time between spikes

    t = 0.0

    spike_times = []
    neuron_ids = []

    # Generate spikes from receptor neurons
    while t <= simulation_time:

        spike_times.append(t)
        neuron_ids.append(current_id)

        current_id += step_dir
        t += time_step

        if current_id < min_id or current_id > max_id: # change direction
            step_dir *= -1
            current_id += 2 * step_dir

    return neuron_ids, spike_times