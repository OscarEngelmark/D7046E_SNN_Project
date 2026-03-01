from spike_generation import linear_movement
import brian2 as b
import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms, mV, Mohm, pA, volt, amp, second
from brian2 import prefs
from matplotlib.ticker import MultipleLocator

# Function to run simulation
def run_simulation(
        num_inputs,
        spike_ids,
        spike_times,
        stdp_enabled=True,
        label='',
        v_rest = -70 * mV,
        v_reset = -80 * mV,
        v_thres = -60 * mV,
        tau_m = 20 * ms,
        r = 100 * Mohm,
        tau_syn = 35 * ms,
        ex_weight = 600 * pA,
        in_weight = -100 * pA,
        lateral_inhib_weight = -20 * pA,
        inhib_delay = 50 * ms,  # for inhibitory groups (not lateral inhibition!)
        tau_pls = 20 * ms,
        tau_mns = 20 * ms,
        gamma = 0.05,  # TUNE ME!
        w_max = 2.0,
        w_min = 0.0,
):

    # Update rules
    eqs_neurons = '''
        dv/dt = ((v_rest - v) + r * I_syn) / tau_m : volt
        dI_syn/dt = -I_syn / tau_syn : amp
    '''

    stdp_eqs = '''
        w : 1
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
        w += gamma * (w - w_max) * post_trace
        w = clip(w, w_min, w_max)
    '''

    b.start_scope()

    # Input group
    input_group = b.SpikeGeneratorGroup(num_inputs, spike_ids, spike_times * second)

    LIF_kwargs = {
        'model': eqs_neurons,
        'threshold': 'v > v_thres',
        'reset': 'v = v_reset',
        'method': 'exact'
    }

    # Left pathway
    left_inhib_group = b.NeuronGroup(N=4, name='left_inhib', **LIF_kwargs)
    left_relay_group = b.NeuronGroup(N=4, name='left_relay', **LIF_kwargs)
    left_output_group = b.NeuronGroup(N=1, name='left_output', **LIF_kwargs)

    # Right pathway
    right_inhib_group = b.NeuronGroup(N=4, name='right_inhib', **LIF_kwargs)
    right_relay_group = b.NeuronGroup(N=4, name='right_relay', **LIF_kwargs)
    right_output_group = b.NeuronGroup(N=1, name='right_output', **LIF_kwargs)

    # Synapses: relay to output
    if stdp_enabled:  # train
        synapse_kwargs = {
            'model': stdp_eqs,
            'on_pre': on_pre_stdp,
            'on_post': on_post_stdp
        }
    else:  # validation/test
        synapse_kwargs = {
            'model': 'w : 1',
            'on_pre': 'I_syn_post += w * ex_weight',
            'on_post': None
        }

    # ────────────────────────────────────────────────
    # Leftward pathway synapses (prefers decreasing IDs: 4 → 3 → 2 → 1 → 0)
    # ────────────────────────────────────────────────

    # Input → left inhib (excitatory, fixed, direct)
    S_input_left_inhib = b.Synapses(input_group, left_inhib_group, model='w : amp', on_pre='I_syn_post += w')
    S_input_left_inhib.connect(i=[0, 1, 2, 3], j=[0, 1, 2, 3])  # inputs 0-3 → left_inhib 0-3
    S_input_left_inhib.w = ex_weight

    # Left inhib → left relay (inhibitory, fixed, matched)
    S_left_inhib_relay = b.Synapses(left_inhib_group, left_relay_group, model='w : amp', on_pre='I_syn_post += w')
    S_left_inhib_relay.connect()  # or 'j == i' — same size → 1:1
    S_left_inhib_relay.w = in_weight
    S_left_inhib_relay.delay = inhib_delay

    # Input → left relay (excitatory, fixed, shifted for leftward preference)
    S_input_left_relay = b.Synapses(input_group, left_relay_group, model='w : amp', on_pre='I_syn_post += w')
    S_input_left_relay.connect(i=[1, 2, 3, 4], j=[0, 1, 2, 3])  # input 1→relay0, 2→1, 3→2, 4→3
    S_input_left_relay.w = ex_weight

    # Left relay → left output (plastic with STDP)
    S_left_relay_output = b.Synapses(left_relay_group, left_output_group, **synapse_kwargs)
    S_left_relay_output.connect()  # all 4 left relays → single left output

    # ────────────────────────────────────────────────
    # Rightward pathway synapses (prefers increasing IDs: 0 → 1 → 2 → 3 → 4)
    # ────────────────────────────────────────────────

    # Input → right inhib (excitatory, fixed, direct)
    S_input_right_inhib = b.Synapses(input_group, right_inhib_group, model='w : amp', on_pre='I_syn_post += w')
    S_input_right_inhib.connect(i=[1, 2, 3, 4], j=[0, 1, 2, 3])  # inputs 1-4 → right_inhib 0-3
    S_input_right_inhib.w = ex_weight

    # Right inhib → right relay (inhibitory, fixed, matched)
    S_right_inhib_relay = b.Synapses(right_inhib_group, right_relay_group, model='w : amp', on_pre='I_syn_post += w')
    S_right_inhib_relay.connect()  # 1:1
    S_right_inhib_relay.w = in_weight
    S_right_inhib_relay.delay = inhib_delay

    # Input → right relay (excitatory, fixed, reversed shift for rightward)
    S_input_right_relay = b.Synapses(input_group, right_relay_group, model='w : amp', on_pre='I_syn_post += w')
    S_input_right_relay.connect(i=[0, 1, 2, 3], j=[0, 1, 2, 3])  # input 0→relay0, 1→1, 2→2, 3→3
    S_input_right_relay.w = ex_weight

    # Right relay → right output (plastic with STDP)
    S_right_relay_output = b.Synapses(right_relay_group, right_output_group, **synapse_kwargs)
    S_right_relay_output.connect()  # all 4 right relays → single right output

    # ────────────────────────────────────────────────
    # Set initial weights for plastic synapses
    # ────────────────────────────────────────────────
    if stdp_enabled:
        # Random initialization for both pathways during training
        S_left_relay_output.w = 'rand() * 0.5'
        S_right_relay_output.w = 'rand() * 0.5'
    else:
        # Load trained weights (you will need two files now)
        left_weights = np.load("left_weights.npy")
        right_weights = np.load("right_weights.npy")
        S_left_relay_output.w[:] = left_weights
        S_right_relay_output.w[:] = right_weights

    # ────────────────────────────────────────────────
    # Lateral (mutual) inhibition between the two outputs
    # ────────────────────────────────────────────────
    S_right_to_left = b.Synapses(right_output_group, left_output_group, model='w : amp', on_pre='I_syn_post += w')
    S_right_to_left.connect()  # single connection
    S_right_to_left.w = lateral_inhib_weight

    S_left_to_right = b.Synapses(left_output_group, right_output_group, model='w : amp', on_pre='I_syn_post += w')
    S_left_to_right.connect()
    S_left_to_right.w = lateral_inhib_weight

    # ────────────────────────────────────────────────
    # Monitors — only what we need for plotting
    # ────────────────────────────────────────────────
    spike_mon_input = b.SpikeMonitor(input_group, name='input_spikes')
    spike_mon_left = b.SpikeMonitor(left_output_group, name='left_output_spikes')
    spike_mon_right = b.SpikeMonitor(right_output_group, name='right_output_spikes')
    state_mon_left = b.StateMonitor(left_output_group, 'v', record=True, name='left_v')
    state_mon_right = b.StateMonitor(right_output_group, 'v', record=True, name='right_v')

    # Run
    sim_duration = max(spike_times) * second + 10 * ms
    b.run(sim_duration)

    print("Left relay weights:", S_left_relay_output.w[:])
    print("Right relay weights:", S_right_relay_output.w[:])

    print(f"Left relay weights mean/std: {np.mean(S_left_relay_output.w):.3f} / {np.std(S_left_relay_output.w):.3f}")
    print(f"Right relay weights mean/std: {np.mean(S_right_relay_output.w):.3f} / {np.std(S_right_relay_output.w):.3f}")

    # ────────────────────────────────────────────────
    # Plotting
    # ────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1.2]})

    # 1. Raster plot: inputs (negative) + left/right outputs
    t_ms_all = spike_mon_input.t / ms  # consistent ms
    input_spike_ids = spike_mon_input.i - num_inputs  # -5 to -1

    left_spike_ms = spike_mon_left.t / ms
    left_spike_ids = np.full_like(left_spike_ms, 0)  # fixed ID 0 for left

    right_spike_ms = spike_mon_right.t / ms
    right_spike_ids = np.full_like(right_spike_ms, 1)  # fixed ID 1 for right

    # Combine
    all_times_ms = np.concatenate([t_ms_all, left_spike_ms, right_spike_ms])
    all_ids = np.concatenate([input_spike_ids, left_spike_ids, right_spike_ids])

    axs[0].scatter(all_times_ms, all_ids, marker='|', s=40, c='k', lw=1.8)
    axs[0].set_ylabel('Neuron ID\n(inputs negative, left output = 0, right = 1)')
    axs[0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0].grid(True, alpha=0.4, linestyle='--')
    axs[0].set_title(f'{label} — Input and Output Spikes')

    # Annotate
    axs[0].text(0.02, 0.98, 'Leftward detector (0)\nRightward detector (1)',
                transform=axs[0].transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 2. Output voltages (both)
    t_ms = state_mon_left.t / ms
    axs[1].plot(t_ms, state_mon_left.v[0] / mV, label='Left output (leftward pref.)', c='darkorange', lw=1.4)
    axs[1].plot(t_ms, state_mon_right.v[0] / mV, label='Right output (rightward pref.)', c='teal', lw=1.4)

    # Threshold
    axs[1].axhline(y=v_thres / mV, color='r', ls='--', lw=1.0, label='threshold')

    # Debug: vlines at output spike times
    for spike_t in left_spike_ms:
        axs[1].axvline(x=spike_t, color='darkorange', ls=':', lw=0.8)
    for spike_t in right_spike_ms:
        axs[1].axvline(x=spike_t, color='teal', ls=':', lw=0.8)

    axs[1].set_ylabel('Membrane potential [mV]')
    axs[1].set_xlabel('Time [ms]')
    axs[1].legend(loc='upper right', fontsize=10)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Stats
    print(f'{label} Stats:')
    print(f'  Left output spikes:  {len(spike_mon_left.i)}')
    print(f'  Right output spikes: {len(spike_mon_right.i)}')

    # Return learned weights (relay to output)
    return np.array(S_left_relay_output.w[:]), np.array(S_right_relay_output.w[:])

def main():
    prefs.codegen.target = "numpy"  # can use cython

    # Number of input receptors
    num_inputs = 5

    params = {
        # Biophysical parameters
        "v_rest": -70 * mV,
        "v_reset": -80 * mV,
        "v_thres": -60 * mV,
        "tau_m": 20 * ms,
        "r": 100 * Mohm,
        "tau_syn": 35 * ms,
        "ex_weight": 600 * pA,
        "in_weight": -100 * pA,
        "lateral_inhib_weight": -20 * pA,
        "inhib_delay": 50 * ms,  # for inhibitory groups (not lateral inhibition!)

        # STDP parameters — pair-based with soft bounds
        "tau_pls": 20 * ms,
        "tau_mns": 20 * ms,
        "gamma": 0.05,  # TUNE ME!
        "w_max": 2.0,
        "w_min": 0.0
    }

    # Simulation durations (in seconds)
    train_time = 10.0
    val_time = 1.5
    test_time = 1.5

    # Speed for motion (neurons/s)
    speed = 20.0

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

    # Run training
    left_weights, right_weights = run_simulation(num_inputs, train_ids, train_times, stdp_enabled=True, label='Training', **params)
    np.save("left_weights.npy", left_weights)
    np.save("right_weights.npy", right_weights)

    # Run validation
    # run_simulation(val_ids, val_times, stdp_enabled=False, label='Validation')

    # Run test
    run_simulation(num_inputs, test_ids, test_times, stdp_enabled=False, label='Test', **params)

if __name__ == '__main__':
    main()