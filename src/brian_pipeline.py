from spike_generation import linear_movement
import brian2 as b
import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms, mV, Mohm, pA, volt, amp, second
from brian2 import prefs
from matplotlib.ticker import MultipleLocator

prefs.codegen.target = "numpy" # can use cython

# Biophysical parameters
v_rest = -70 * mV
v_reset = -80 * mV
v_thres = -55 * mV
tau_m = 20 * ms
R = 100 * Mohm
tau_syn = 39 * ms
ex_weight = 350 * pA
in_weight = -250 * pA

# Number of input receptors
num_inputs = 5

# Simulation durations (in seconds)
train_time = 100.0
val_time = 3.0
test_time = 3.0

# Speed for motion (neurons/s)
speed = 18.0

# Generate datasets
train_ids, train_times = linear_movement(num_inputs, train_time, speed)
val_ids, val_times = linear_movement(num_inputs, val_time, speed)
test_ids, test_times = linear_movement(num_inputs, test_time, speed)

# Convert to numpy
train_ids = np.array(train_ids)
train_times = np.array(train_times)
val_ids = np.array(val_ids)
val_times = np.array(val_times)
test_ids = np.array(test_ids)
test_times = np.array(test_times)

# Neuron model (LIF with exponential current synapses)
eqs_neurons = '''
dv/dt = ((v_rest - v) + R * I_syn) / tau_m : volt
dI_syn/dt = -I_syn / tau_syn : amp
'''

# STDP parameters — pair-based with soft bounds
tau_pls   = 20 * ms
tau_mns   = 20 * ms
gamma     = 1.0 # TUNE ME!
w_max     = 2.0
w_min     = 0.0

stdp_eqs = '''
w     : 1
pre_trace : 1
post_trace : 1
'''

on_pre_stdp = '''
I_syn_post += w * ex_weight
pre_trace += 1
w += gamma * (w_max - w) * pre_trace
w = clip(w, w_min, w_max)
'''

on_post_stdp = '''
post_trace += 1
w += gamma * (w_min - w) * post_trace
w = clip(w, w_min, w_max)
'''

# Function to run simulation
def run_simulation(spike_ids, spike_times, stdp_enabled=True, label=''):
    b.start_scope()

    # Input group
    input_group = b.SpikeGeneratorGroup(num_inputs, spike_ids, spike_times * second)

    # Inhibitory group (4 neurons)
    inhib_group = b.NeuronGroup(4, eqs_neurons, threshold='v > v_thres', reset='v = v_reset', method='exact')

    # Relay group (4 neurons)
    relay_group = b.NeuronGroup(4, eqs_neurons, threshold='v > v_thres', reset='v = v_reset', method='exact')

    # Output group (1 neuron)
    output_group = b.NeuronGroup(1, eqs_neurons, threshold='v > v_thres', reset='v = v_reset', method='exact')

    # Synapses: input to inhib (excitatory, fixed)
    S_input_inhib = b.Synapses(input_group, inhib_group, 'w : amp', on_pre='I_syn_post += w')
    S_input_inhib.connect(i=[0,1,2,3], j=[0,1,2,3])  # Inputs 0-3 to inhib 0-3
    S_input_inhib.w = ex_weight

    # Synapses: inhib to relay (inhibitory, fixed)
    S_inhib_relay = b.Synapses(inhib_group, relay_group, 'w : amp', on_pre='I_syn_post += w')
    S_inhib_relay.connect(j='i')  # Inhib 0-3 to relay 0-3
    S_inhib_relay.w = in_weight

    # Synapses: input to relay (excitatory, fixed, shifted)
    S_input_relay = b.Synapses(input_group, relay_group, 'w : amp', on_pre='I_syn_post += w')
    S_input_relay.connect(i=[1,2,3,4], j=[0,1,2,3])  # Inputs 1-4 to relay 0-3
    S_input_relay.w = ex_weight

    # Synapses: relay to output
    if stdp_enabled: # train
        syn_model = stdp_eqs
        syn_on_pre = on_pre_stdp
        syn_on_post = on_post_stdp
    else: # validation/test
        syn_model = 'w : 1'
        syn_on_pre = 'I_syn_post += w * ex_weight'
        syn_on_post = None

    S_relay_output = b.Synapses(relay_group, output_group,
                                model=syn_model,
                                on_pre=syn_on_pre,
                                on_post=syn_on_post if syn_on_post is not None else None)
    S_relay_output.connect()

    if stdp_enabled:
        S_relay_output.w = 'rand() * 0.5'  # random init
    else:
        loaded_weights = np.load("trained_relay_weights.npy")
        S_relay_output.w[:] = loaded_weights

    # Monitors
    spike_mon_input = b.SpikeMonitor(input_group)
    spike_mon_inhib = b.SpikeMonitor(inhib_group)
    spike_mon_relay = b.SpikeMonitor(relay_group)
    spike_mon_output = b.SpikeMonitor(output_group)
    state_mon_output = b.StateMonitor(output_group, 'v', record=True)

    # Run
    sim_duration = max(spike_times) * second + 10 * ms
    b.run(sim_duration)

    # Plotting (similar to attached code)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 0.8]})

    # Raster plot (all spikes: inputs negative, then inhib, relay, output)
    all_spike_times = np.concatenate([spike_mon_input.t/second, spike_mon_inhib.t/second,
                                      spike_mon_relay.t/second, spike_mon_output.t/second])
    all_spike_ids = np.concatenate([spike_mon_input.i - num_inputs,  # Negative for inputs
                                    spike_mon_inhib.i,
                                    spike_mon_relay.i + 4,  # Offset
                                    spike_mon_output.i + 8])  # Output at top
    axs[0].scatter(all_spike_times, all_spike_ids, marker='|', s=20, c='k', lw=1.8)
    axs[0].set_ylabel("Neuron ID (inputs negative, inhib 0-3, relay 4-7, output 8)")
    axs[0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0].grid(True, alpha=0.5)
    axs[0].set_title(f'{label} Simulation')

    # Output voltage
    axs[1].plot(state_mon_output.t/second, state_mon_output.v[0]/mV, label="Output", c='darkblue')
    axs[1].axhline(y=v_thres/mV, color='r', ls='--', lw=0.9, label="threshold")
    axs[1].set_ylabel("u_output [mV]")
    axs[1].set_xlabel("Time [s]")
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Stats
    print(f'{label} - Output spikes: {len(spike_mon_output.i)}')

    # Return learned weights (relay to output)
    return np.array(S_relay_output.w[:])

# Run training
train_weights = run_simulation(train_ids, train_times, stdp_enabled=True, label='Training')
np.save("trained_relay_weights.npy", train_weights)

# Run validation
run_simulation(val_ids, val_times, stdp_enabled=False, label='Validation')

# Run test
run_simulation(test_ids, test_times, stdp_enabled=False, label='Test')