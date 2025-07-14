import os, gc
import numpy as np; np.set_printoptions(legacy='1.25')
import scipy
from pathlib import Path

from oscillations.utils import circ_rtest, pairwise_phase_consist

folderW = 'output'
rootFolder = os.getcwd()

filtOrder = 4

plot_by_cell = False

def compute_phase_coding(regions, spikes, long_signal, epoched_signal, watch_depths, filtFreq, lfp_sampling_rate, compute_PPC, side=0, plot_per_cell=False):
    import xarray as xr

    import time as TIME
    start_time = TIME.time()

    compute_for_trial_in_total = False

    if watch_depths is None:
        watch_depths = epoched_signal.depth.values[::2] # take every second one (for either side)

    reg_spikes, reg_lfp = regions.split('-')

    all_cells = np.unique(spikes['cell_id'].values)

    task_stages = spikes.attrs['task_stage_names'] # e.g. ['before', 'unlcok', 'tone', 'after']
    if compute_for_trial_in_total:
        task_stages.append('all')

    final_data_shape = (len(all_cells), len(watch_depths), len(task_stages), 7) # depths (dim 1) are from deeper to superficial; last dim is (r,theta, pval1, pval2, z, num_spikes, PPC)

    result = np.zeros(final_data_shape)

    print(f'Will compute association from {reg_spikes} spikes to {reg_lfp} LFP in band {filtFreq[0]}-{filtFreq[1]}, shape {result.shape}')

    # Filtering each LFP to analize

    for iDepth, depth in enumerate(watch_depths): # nk goes from bottom to top (e.g. 1680 to 20)
        print(f'Watch depth in {reg_lfp}: {depth}')

        # indices for given depths (two values - for two sides)
        channel_ind = epoched_signal.chan.where(epoched_signal.depth==depth, drop=True).values

        assert channel_ind.size > 0, f"No such depth {depth}!"
        channel_ind = channel_ind.astype(int)[side]  # using one side (todo: try another)

        rawSignal = long_signal[channel_ind]

        fs = lfp_sampling_rate
        time = np.linspace(0,len(rawSignal)*1000/fs,len(rawSignal))   # in milliseconds

        # np.concatenate((rawSignal, [2*rawSignal[-1]-rawSignal[-2]])) # extrapolation for the last element
        rawSignal = scipy.stats.zscore(np.float32(rawSignal))

        # from oscillations.utils import surrogate_of
        # surrog = surrogate_of(rawSignal, lfp_sampling_rate)
        # rawSignal = surrog
        rawSignal_ = np.r_[rawSignal[-1::-1],rawSignal,rawSignal[-1::-1]] # prepend and append reflected signal to minimize edge artifacts

        nyquist = fs/2.0

        Wn = [filtFreq[0]/nyquist, filtFreq[1]/nyquist]
        b, a = scipy.signal.butter(filtOrder, Wn, btype='bandpass')

        rawSignalFiltered_ = scipy.signal.filtfilt(b, a, rawSignal_)

        analytic_signal = scipy.signal.hilbert(rawSignalFiltered_)[len(rawSignal):-len(rawSignal)]
        # amplitude_envelope = np.abs(analytic_signal)

        # instantaneous_phase = np.angle(analytic_signal) * 180 / pi
        instantaneous_phase = np.unwrap(np.angle(analytic_signal)) # unwrap before interpolating
        f_Rhythm = scipy.interpolate.interp1d(time, instantaneous_phase)

        phases = f_Rhythm(spikes['spike_time'] * 1000)
        phases = (np.mod(phases + np.pi, 2*np.pi) - np.pi)

        # Association of cell spiking to filtered signals
        for icell, cell_gid in enumerate(all_cells):

            # Statistics
            # Collecting spikes for each task stage
            for stage, st_name in enumerate(task_stages):
                this_cell_all_spikes_mask = spikes['cell_id'] == cell_gid
                if st_name == 'all': # whole trial
                    alpha = phases[this_cell_all_spikes_mask]
                else:
                    alpha = phases[this_cell_all_spikes_mask & (spikes['stage'] == stage)]
                
                if len(alpha) == 0:
                    print(f"No spikes for cell {cell_gid} on stage '{task_stages[stage]}'")
                    result[icell, iDepth, stage] = (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)
                    continue

                values = circ_rtest(alpha)
                r, theta, pval1, pval2, z = values
                PPC = pairwise_phase_consist(alpha) if compute_PPC else np.nan
                result[icell, iDepth, stage] = (r, theta, pval1, pval2, z, len(alpha), PPC)

            if plot_per_cell:
                trial_window = spikes.attrs.get('trial_window')
                assert trial_window, "When plot_by_cell is True, trial_window should be provided"
                from oscillations.plot_spike_lfp_coher import plot_for_cell
                plot_for_cell(
                    cell_gid,
                    spikes[this_cell_all_spikes_mask],
                    epoched_signal.sel(chan=channel_ind),
                    phases[this_cell_all_spikes_mask],
                    result[icell, iDepth],
                    reg_spikes, reg_lfp, depth, trial_window, rootFolder, folderW)

        gc.collect() # clean-up after each depth

    runtime = TIME.time() - start_time

    band = f'{filtFreq[0]}-{filtFreq[1]}'
    folder = 'data/spike-field-coherence'
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f'{folder}/{regions}_{band}_side{side}'
    print(f'Done in {runtime}. Saving as {filename}')

    np.save(filename + '.npy', result)

    # create final data structure
    coords = {
        'cell': all_cells, # ('cell', 
        'cell_depth': ('cell', spikes.attrs['cell_depths'][all_cells]),
        'depth': watch_depths,
        'stage': task_stages,
        'value': ['r', 'theta', 'pval1', 'pval2', 'z', 'num_spikes', 'PPC']
    }
    result = xr.DataArray(data=result,
                          dims=['cell', 'depth', 'stage', 'value'],
                          coords=coords
                          )
    result.to_netcdf(filename + '.nc', engine='h5netcdf', invalid_netcdf=True)

def run(folder, regspikes, reglfp, depths, freq=[8, 12]):

    from  preprocess.epoch_data import load_lfp, load_epoched_spikes, load_epoched_data
    sess_dir_path = Path(f'./data/{folder}')

    # Load epoched data
    X = load_epoched_data(sess_dir_path, reglfp)
    spikes = load_epoched_spikes(sess_dir_path, regspikes)

    # LFP or Joystick data
    x = X.LFP
    # x = X.Joystick

    raw_sig = load_lfp(sess_dir_path, reglfp)[f'{reglfp}_lfps'].T
    compute_phase_coding(f'{regspikes}-{reglfp}', spikes, raw_sig, x, depths, freq, x.attrs['fs'], compute_PPC=False, plot_per_cell=False)

if __name__ == '__main__':
    run('230517_2759_1606VAL', regspikes='m1', reglfp='m1', depths=[1680, 1660])
    run('230517_2759_1606VAL', regspikes='m1', reglfp='th', depths=[3820, 3800])
    run('230517_2759_1606VAL', regspikes='th', reglfp='m1', depths=[1680, 1660])
    run('230517_2759_1606VAL', regspikes='th', reglfp='th', depths=[3820, 3800])