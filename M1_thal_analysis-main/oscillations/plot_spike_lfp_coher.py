import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os, pickle as pkl



def plot_for_cell(ncell, spikes, signal, spike_phases, tests_result, reg_name_spikes, reg_name_lfp, depth_val, trial_window, rootFolder, folderW):

    n_trials = len(signal)
    trial_inds = spikes['trial'].values

    times = spikes['spike_time'] # absolute spike times
    times -= signal.sel(trial=trial_inds).tone_onset.values # relative to tone onset

    levels = trial_inds
    colors = spike_phases[spikes.index]

    fig = plt.figure(dpi=150)
    ax1 = plt.subplot2grid((6,4), (0,0), rowspan=3, colspan=2)
    ax2 = plt.subplot2grid((6,4), (0,2), rowspan=3, colspan=2)
    ax3 = plt.subplot2grid((6,4), (3,0), projection='polar', rowspan=2)
    ax4 = plt.subplot2grid((6,4), (3,1), projection='polar', rowspan=2)
    ax5 = plt.subplot2grid((6,4), (3,2), projection='polar', rowspan=2)
    ax6 = plt.subplot2grid((6,4), (3,3), projection='polar', rowspan=2)
    ax7 = plt.subplot2grid((6,4), (5,0))
    ax8 = plt.subplot2grid((6,4), (5,1))
    ax9 = plt.subplot2grid((6,4), (5,2))
    ax10 = plt.subplot2grid((6,4), (5,3))

    fig.set_figheight(15)
    fig.set_figwidth(10)
    fig.suptitle(f'Cell {ncell+1} (Area {reg_name_spikes}) - Depth {depth_val} (Area {reg_name_lfp})', fontsize=12)

    # spikes
    ax1.scatter(times, levels, marker="|", s=20, linestyle='None', c = colors, cmap='hsv')
    ax1.set_xlim([trial_window[0], trial_window[1]])
    ax1.set_ylim([-1, n_trials])
    ax1.vlines(0, -1, n_trials, linestyle='--', color='k')
    ax1.set_ylabel('Trial #', fontsize=12)
    ax1.set_xlabel('Time (s)', fontsize=12)

    # LFP
    # ax2.set_xlim([-DeltaT_pre, DeltaT_post])
    ax2.set_ylim([-1, n_trials])
    ax2.vlines(0, -1, n_trials, linestyle='--', color='k')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_yticks([])

    for ntrial, trial in enumerate(signal):
        # highlight task phases (unlock and tone) on both spikes and signal panels
        unlock_time_rel = trial.unlock - trial.tone_onset
        toneoff_time_rel = trial.tone_offset - trial.tone_onset

        ax1.hlines(ntrial, unlock_time_rel, 0, linestyle='-', linewidths=4, color=(1,0,0,0.25))
        ax2.hlines(ntrial, unlock_time_rel, 0, linestyle='-', linewidths=4, color=(1,0,0,0.25))
        ax1.hlines(ntrial, 0, toneoff_time_rel, linestyle='-', linewidths=4, color=(0.25,0.25,0.25,0.25))
        ax2.hlines(ntrial, 0, toneoff_time_rel, linestyle='-', linewidths=4, color=(0.25,0.25,0.25,0.25))

        # plot signal per trial
        sig = signal.sel(time=slice(trial_window[0], trial_window[1]), trial=ntrial)
        norm = 10
        ax2.plot(sig.time, sig.values / norm + ntrial, color=(0.5,0.5,0.5,1))

    # Redefining axes 1 to accomodate colorbar 
    ax1.set_position([ax1.get_position().x0, ax1.get_position().y0, (ax1.get_position().x1-ax1.get_position().x0), 0.8*(ax1.get_position().y1-ax1.get_position().y0)])
    ax2.set_position([ax2.get_position().x0, ax2.get_position().y0, (ax2.get_position().x1-ax2.get_position().x0), 0.8*(ax2.get_position().y1-ax2.get_position().y0)])

    cax = fig.add_axes([ax1.get_position().x0 + 0.125*(ax1.get_position().x1-ax1.get_position().x0), ax1.get_position().y1+0.1*(ax1.get_position().y1-ax1.get_position().y0), 0.75*(ax1.get_position().x1-ax1.get_position().x0), 0.02])
    import matplotlib as mpl
    cbar = fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-180, vmax=180), cmap='hsv'), cax=cax, ticks=[-180, -90, 0, 90, 180],orientation='horizontal')
    cbar.set_label('Phase', rotation=0)

    def plot_circ(spks, values, ax, ax_lhist, text_x, label):
        from oscillations.utils import circular_hist
        r, theta, pval1, pval2, z, _, _ = values
        circular_hist(ax, spks, bins=18)
        ax.text(text_x, -0.2, f'{label} (n = {len(spks)})', transform=ax.transAxes, fontsize=12, verticalalignment='top') 
        ax.text(text_x, -0.3, f'  r = {r:.3}', transform=ax.transAxes, fontsize=12, verticalalignment='top') 
        ax.text(text_x, -0.4, f'  theta = {np.round(theta*100)/100}', transform=ax.transAxes, fontsize=12, verticalalignment='top') 
        ax.text(text_x, -0.5, f'  z = {z:.3}', transform=ax.transAxes, fontsize=12, verticalalignment='top') 
        ax.text(text_x, -0.6, f'  p = {pval1:.3}', transform=ax.transAxes, fontsize=12, verticalalignment='top') 

        ax_lhist.hist(x=spks*180/np.pi, bins=18, range=(-180,180))
        ax_lhist.set_xticks([-180,-90,0,90,180])

    plot_circ(spike_phases[spikes['stage'] == 0], tests_result[0, :], ax3, ax7, 0.2, 'Before')
    plot_circ(spike_phases[spikes['stage'] == 1], tests_result[1, :], ax4, ax8, 0.3, 'Pre')
    plot_circ(spike_phases[spikes['stage'] == 2], tests_result[2, :], ax5, ax9, 0.15, 'Tone ON')
    plot_circ(spike_phases[spikes['stage'] == 3], tests_result[3, :], ax6, ax10, 0.225, 'After')

    folder = os.path.join(rootFolder, folderW, f'{reg_name_spikes}-cell_{ncell+1}')
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, f'Association_{reg_name_spikes}_Cell_{ncell+1}_LFP({reg_name_lfp})Depth{depth_val}.png')
    plt.savefig(filename)
    plt.close(fig)


output_folder = 'output/spike-to-lfp'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def cell_color(i):
    colors = plt.get_cmap('Dark2').colors[:-1] + \
        plt.get_cmap('Set1').colors[:-1] + \
        plt.get_cmap('Set2').colors[:-1] # drop last elements which are grayish
    return colors[i % len(colors)]


MOI_PLV, MOI_PPC, MOI_RAYL_Z = 0, 6, 4
moi_titles = {MOI_PLV: 'Tot. Phase Locking Value', MOI_PPC: 'Tot. Pairwise Phase Consistency', MOI_RAYL_Z: 'Avg. Rayleigh Z'}


# spikes in M1
with open (os.path.join('data/230517_2759_1606VAL', 'm1spts.pkl'), 'rb') as f:
    m1_spks = pkl.load(f)
    cell_depths = m1_spks['depths_from_surface']

def plot_assoc(assoc_m1_m1, moi_ind, moi_thresh, survived, figtitle, ax=None):
    if not ax:
        plt.figure()
        ax = plt

    # assoc_m1_m1 = assoc_m1_m1[:,::-1,:] # invert depth

    nDepths = assoc_m1_m1.shape[1]
    depths = np.linspace(1680, 20, nDepths) / 1000

    p_thresh = 0.05
    # r_thresh = 0.25 # np.percentile(assoc_m1_m1[:,:,0], 66) # plotting best third of cells
    for nc, c in enumerate(assoc_m1_m1):
        if nc in survived: # mean_r > r_thresh:
            p = assoc_m1_m1[nc,:,2]
            p_good = p < p_thresh
            p_bad = p >= p_thresh

            moi = assoc_m1_m1[nc,:, moi_ind] # e.g. PLV, PPC or Rayleigh z
            theta = assoc_m1_m1[nc,:,1]
            r = assoc_m1_m1[nc,:,MOI_PPC] # MOI_PLV

            try:
                for i in range(len(theta)*2):
                    diff = np.diff(theta)
                    if diff.max() > 180:
                        theta[np.argmax(diff)+1:] -= 360
                    elif diff.min() < -180:
                        theta[np.argmin(diff)+1:] += 360
                    else:
                        break
                if theta.max() > 360:
                    theta -= 360
                if theta.min() < -360:
                    theta += 360
            except:
                pass

            cell_col = cell_color(nc)

            minval, maxval = moi_thresh, 1
            minlwidth, widthrange = 0.2, 3
            val = moi.mean() # could also be sum
            lwidth = minlwidth + widthrange * (val - minval) / (maxval - minval)
            
            scatter = False
            if scatter:
                marker_scale = 10
                # ax.scatter(theta[p_good], depths[p_good], color=cell_col, s= r[p_good] * marker_scale)
                ax.scatter(theta[p_bad], depths[p_bad], edgecolors=cell_col, facecolors='none', s= moi[p_bad] * marker_scale)
            plot = False
            if plot:
                ax.plot(theta[p_good], depths[p_good], linewidth=lwidth, color=cell_col) # , color='b',

            # plotting r
            ax.plot(r, depths, linewidth=lwidth, color=cell_col) # , color='b',
            ax.scatter(r[p_good], depths[p_good], edgecolors=cell_col, facecolors='none') # , s= moi[p_bad] * marker_scale
            # ax.set_xlim(0, 1)

            # plot cell depth bar
            ax.hlines(cell_depths[nc]/1000, 0.95, 1, colors=cell_col, transform=ax.transAxes)
            print(f'{nc} -- {i}')
    ax.invert_yaxis()
    ax.set_ylabel('LFP depth (mm)')
    # plt.title(figtitle)

def plot_per_band(assocs, titles, band, fig_title):
    # assocs shape  (r, theta, pval1, pval2, z, num_spikes)

    band_name = f'_{band[0]}-{band[1]}Hz'

    fig = plt.figure(figsize=(10,12), dpi=300)
    gridDim = (len(assocs) * 2, 2)
    # fig, axs = plt.subplots(2 * len(assocs), figsize=(10, 15), sharex=True)

    survived_cells = [{}] * len(assocs)
    survived_pvalue = []
    survived_r = []

    moi_ind = MOI_PLV

    for ind, assoc in enumerate(assocs):

        # survived_cells

        nCell = len(assoc)
        moi = assoc[:,:,moi_ind]
        p = assoc[:,:,2]
        mean_p = assoc[:,:,2].mean(axis=1)
        p_tresh = 0.1
        moi_thresh = 0.0

        # plot r
        ax = plt.subplot2grid(gridDim, (ind * 2 + 0, 0), colspan=1)
        ax.set_title(titles[ind])
        ax.set_ylabel(moi_titles[moi_ind])
        ax.set_xlim([0, nCell])
        # ax.set_ylim([0, 10])

        survived_pvalue.append([])
        survived_r.append([])
        for iCell, radPerDepth in enumerate(moi):
            if mean_p[iCell] < p_tresh:
                survived_pvalue[ind].append(iCell)

                mean_moi = radPerDepth.mean() # could also be sum

                if mean_moi > moi_thresh:
                    survived_r[ind].append(iCell)

                    color = cell_color(iCell)
                else:
                    color = 'gray'

                ax.scatter(iCell * np.ones_like(radPerDepth), radPerDepth, color=color, marker='o') # , facecolors='none', edgecolors='red'
                ax.scatter(iCell, mean_moi, marker='*', color='k')

        # plot p-value
        ax = plt.subplot2grid(gridDim, (ind * 2 + 1, 0), colspan=1, sharex=ax)
        ax.set_ylabel('p-value')
        ax.set_xlim([0, nCell])
        ax.set_ylim([0, 0.4])
        # ax.set_yscale('log')
        ax_cells = ax

        for iCell, pPerDepth in enumerate(p):
            meanP = mean_p[iCell]
            if meanP < p_tresh:
                ax.scatter(iCell * np.ones_like(pPerDepth), pPerDepth, color=cell_color(iCell), marker='.')
                ax.scatter(iCell, meanP, color='k', marker='o') # , facecolors='none', edgecolors='red'
            else:
                ax.scatter(iCell * np.ones_like(pPerDepth), pPerDepth, color='gray', marker='.', s=0.5)
                ax.scatter(iCell, meanP, marker='o', facecolors='none', edgecolors='gray')
        # ax.sharex(ax0)

        ax_phase = plt.subplot2grid(gridDim, (ind * 2, 1), rowspan=2)
        plot_assoc(assoc, moi_ind, moi_thresh, survived_r[ind], titles[ind], ax_phase)

    ax_phase.set_xlabel('Phase (°)')
    ax_cells.set_xlabel('Cell #')

    plt.subplots_adjust(hspace=0.7)
    # fig.set_size_inches(10, 12)

    plt.savefig(f'{output_folder}/{fig_title}{band_name}.png') # 'joy_
    # for i, meanP in enumerate(mean_p):
    #     if mean_p[i] < p_tresh:
    #         axs[2].scatter(i, meanP, marker='o') # , facecolors='none', edgecolors='red'
    #     else:
    #         axs[2].scatter(i, meanP, marker='o', facecolors='none', edgecolors='gray')

    # for i, c in enumerate(assoc):
    #     radii = c[:,4]
    #     axs[3].scatter(i * np.ones_like(radii), radii, marker='o') # , facecolors='none', edgecolors='red'
    # for i, c in enumerate(assoc):
    #     radii = c[:,5]
    #     axs[4].scatter(i * np.ones_like(radii), radii, marker='o') # , facecolors='none', edgecolors='red'
# meanrs[meanrs>0.8]

def load(band, files_label):
    ndep = '' # '_ndep84'
    band = f'_{band[0]}-{band[1]}'

    folder = 'data/assoc/'
    regs = files_label

    # assoc_all = np.load(f'{folder}{regs}_all{ndep}{band}.npy')
    assoc_before = np.load(f'{folder}{regs}_before{ndep}{band}.npy')
    assoc_pre = np.load(f'{folder}{regs}_pre{ndep}{band}.npy')
    assoc_toneon = np.load(f'{folder}{regs}_toneon{ndep}{band}.npy')
    assoc_post = np.load(f'{folder}{regs}_post{ndep}{band}.npy')

    return [assoc_before, assoc_pre, assoc_toneon, assoc_post], ['Before', 'Pre', 'Tone on', 'Post']

def plot_all_avg(bands, moi, plot_per_band, files_label, save_fig_title=None, color='tab:blue', figAndAxs=None, grand_max_r=0, grand_min_r=0):

    n_periods = 4
    n_bands = len(bands)
    depths = np.linspace(1680, 20, 84) # TODO: de-hardcode

    p_thresh = 1 if moi == MOI_PPC else 0.2 # for PPC, accept all
    
    min_spikes_quant = None # 0.75
    num_spikes_thresh = 10

    if figAndAxs is None:
        fig, axs = plt.subplots(n_bands if plot_per_band else 1, n_periods) # 5 periods per each band
    else:
        fig, axs = figAndAxs
    fig.set_size_inches(14, 12 if plot_per_band else 8)
    fig.set_dpi(400)

    if not plot_per_band:
        # values = np.zeros((n_periods, n_bands))
        values = np.zeros((n_periods, n_bands, len(depths)))

    for iBand, band in enumerate(bands): # each `band` is [from, to]
        band_title = f'{band[0]} to {band[1]} Hz'
        data_per_period, period_titles = load(band, files_label)

        
        # total r per periodS

        surv_cells_per_period = []
        sp_thresh_per_period = []

        for iPeriod, assoc in enumerate(data_per_period):
            p_thresh_inds = assoc[:,:,2].mean(axis=1) < p_thresh # mean across depths
            survived_p_cells = assoc[p_thresh_inds]

            if min_spikes_quant:
                num_spikes_thresh = np.quantile(survived_p_cells[:,0,5], min_spikes_quant)
                # num_spikes_thresh = max(1, quantile)

            if moi == MOI_PPC: # thresholding by number of spikes
                thresh_inds = survived_p_cells[:,0,5] > num_spikes_thresh # `5` in third dim is num spikes. Second dim is depth, but num spikes is the same accross all depths, so using 0th depth
            else:
                thresh_inds = survived_p_cells[:,:,0].mean(axis=1) > 0.25 # mean PLV (across depths) to be higher than threshold # TODO: use squared?

            survived_cells = survived_p_cells[thresh_inds]
            surv_cells_per_period.append(len(survived_cells))
            sp_thresh_per_period.append(int(num_spikes_thresh))

            moi_per_depth = survived_cells[:,:,moi].sum(axis=0) # sum across cells
            if moi == MOI_PPC:
                # normalize
                moi_per_depth /= len(survived_cells)

            if plot_per_band:
                # if plot_per_band, each band has axis where r is plotted for all depths

                ax = axs[n_bands-1-iBand, iPeriod] # higher freq bands are on the top

                old_joyst = len(moi_per_depth) == 2 and moi_per_depth[0] == moi_per_depth[1] # hack for old joyst data (where it was doubled)
                if len(moi_per_depth) == 1 or old_joyst:
                    ax.axvline(moi_per_depth[0], c=color)
                else:
                    ax.plot(moi_per_depth, depths / 1000, color=color)
                    ax.invert_yaxis()
                    ax.axvline(moi_per_depth.mean(), c=color)
                # # zero:
                # ax.axvline(0, c='tab:red', linestyle='--')

                if iBand == 0:
                    ax.set_title(f"{period_titles[iPeriod]} ({surv_cells_per_period[iPeriod]} c {sp_thresh_per_period[iPeriod]} sp")
                if iBand == len(bands) - 1:
                    ax.set_xlabel('Avg. PPC') # (f'Total R across\n signif. cells') # {len(survived_r)}
                if iPeriod == 0:
                    ax.set_ylabel(f'Freq: {band_title}.\nLFP Depths (mm):')

                max_r = moi_per_depth.max()
                if max_r > grand_max_r:
                    grand_max_r = max_r
                min_r = moi_per_depth.min()
                if min_r < grand_min_r:
                    grand_min_r = min_r
            else:
                # average across depths
                # val =  moi_per_depth.mean()
                values[iPeriod, iBand] = moi_per_depth
                # if val > grand_max_r:
                #     grand_max_r = val
                # if val < grand_min_r:
                #     grand_min_r = val

    # if plot_per_band:
    #     for iRow, row in enumerate(axs):
    #         # axs[iRow].set_xlim([grand_min_r*1.1, grand_max_r*1.1])
    #         for iCol in range(len(row)):
    #             axs[iRow, iCol].set_xlim([grand_min_r*1.1, grand_max_r*1.1])
    # else:
    #     for iAx, ax in enumerate(axs):
    #         ax.set_xlim([grand_min_r*1.1, grand_max_r*1.1])


    if not plot_per_band:
        freqs = [np.mean(b) for b in bands]
        for iPeriod in range(n_periods):
            ax = axs[iPeriod]
            # ax.plot(values[iPeriod, :], freqs, color=color)
            # # ax.scatter(values[iPeriod, :], freqs, color=color, marker='x')
            # ax.set_xlabel('Avg. PPC')
            # ax.set_ylabel('Freq. Hz')

            extent = [freqs[0], freqs[-1], depths[-1]/1000, depths[0]/1000] # xmin, xmax, ymin, ymax
            img = ax.imshow(values[iPeriod, :,:].T, vmin=-0.02, vmax=0.05, extent=extent, aspect='auto')
            ax.invert_yaxis()
            plt.colorbar(img, ax=ax, orientation = 'horizontal', label='Avg. PPC')
            ax.set_xlabel('Freq. (Hz)')
            ax.set_ylabel('LFP Depth (mm)')
            
           
            ax.set_title(f"{period_titles[iPeriod]}\n({surv_cells_per_period[iPeriod]} cells, at least {sp_thresh_per_period[iPeriod]} spikes)")
            
    
    # for ax in axs.flatten():
    #     grand_max_r = min(0.02, grand_max_r)
    #     ax.set_xlim([grand_min_r*1.1, grand_max_r*1.1])

    plt.subplots_adjust(wspace=0.4)

    if save_fig_title:
        sp = f"Q{min_spikes_quant}" if min_spikes_quant else f"{num_spikes_thresh}"
        fig.savefig(f'{output_folder}/{save_fig_title}_sp{sp}{"" if plot_per_band else "_avgDep"}.png')
    return fig, axs, grand_max_r, grand_min_r


        #     tot_r_across_cells_per_depth = assoc[survived_r[ind],:, 0].sum(axis=0)
        #     total_r_per_period.append(tot_r_across_cells_per_depth)
        #     if tot_r_across_cells_per_depth.max() > max_r:
        #         max_r = tot_r_across_cells_per_depth.max()
        # for i in range(len(data_per_period)):
        #     ax = plt.subplot2grid(gridDim, (i * 2, 2), rowspan=2)
        #     ax.set_xlim([0, max_r])
        #     tot_r = total_r_per_period[i]
        #     plt.plot(tot_r, np.arange(len(tot_r)))
        # ax.set_xlabel('Total r')

def plot_ppc_vs_numspikes(data_per_period, titles):
    plt.figure()
    
    for data in [data_per_period[1]]:
        dep = 0
        data = data[data[:,dep,6]>-2,:,:]
        plt.scatter(data[:,dep,5], data[:,dep,6], marker='.', label='PPC')
        plt.scatter(data[:,dep,5], data[:,dep,0], marker='.', label='PLV squared')
        plt.axhline(0, linestyle='--', c='red')
    plt.xscale('log')
    plt.xlabel('value')
    plt.ylabel('# spikes')
    plt.legend()
    plt.show()

def plot_cells_rates(data_per_period, titles):

    periodDur = np.array([61.533, 82.499, 34.685, 73.795])
    numPeriods = len(data_per_period)
    numCells = data_per_period[0].shape[0]
    rates = np.zeros((numPeriods, numCells))

    for iPeriod in range(numPeriods):
        rates[iPeriod] = data_per_period[iPeriod][:, 0, 5] / periodDur[iPeriod]

    normalized_depths = (cell_depths - np.min(cell_depths)) / (np.max(cell_depths) - np.min(cell_depths))
    cmap = plt.get_cmap('winter').reversed()

    fig, ax = plt.subplots()
    for iCell in range(92):
        ratesPerPeriod = rates[:,iCell]
        zscores = ratesPerPeriod # stats.zscore(ratesPerPeriod)
        # plt.plot(np.arange(numPeriods), zscores)
        col = cmap(normalized_depths[iCell])
        plt.scatter(np.argmax(zscores), zscores.max(), color=col)
        plt.plot([np.argmax(zscores), np.argmin(zscores)], [zscores.max(), zscores.min()], linewidth=0.5, color=col)
    plt.ylabel('Firing rate')
    ax.set_xticks(range(len(titles)), labels=titles)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.max(cell_depths), vmax=np.min(cell_depths)))
    sm.set_array([])
    plt.colorbar(sm, label='Cell depth', ax=ax)

def plot_cells_rates_depth(spikes, task_events, trial_window):
    depths = spikes.attrs['cell_depths']
    stage_names = spikes.attrs['task_stage_names']
    grouped = spikes.groupby('cell_id')
    cells = np.fromiter(grouped.groups.keys(), dtype=int)
    depthsorted_cells = sorted(cells, key=lambda x: depths[x])
    # num_spikes = grouped.size()[depthsorted_cells].values

    spikes_count_by_cell_by_stage = spikes.pivot_table(index='cell_id', columns='stage', values='spike_time', aggfunc='count').fillna(0).astype(int)

    # mean duration of each stage
    dur_unlock = task_events['tone_onset'] - task_events['unlock']
    dur_tone = task_events['tone_offset'] - task_events['tone_onset']
    dur_before = -trial_window[0] - dur_unlock
    dur_after = trial_window[1] - dur_tone

    rates = spikes_count_by_cell_by_stage / [dur_before.sum(), dur_unlock.sum(), dur_tone.sum(), dur_after.sum()]

    means = rates.mean(axis=1)
    stds = rates.std(axis=1)

    rates_znorm = rates.sub(means, axis=0).div(stds, axis=0)

    def sort_fun(row):
        idc = row.sort_values().index
        return np.array([idc[-2], idc[-1]])
    srtd = rates_znorm.apply(lambda row: row.sort_values().index[[-2, -1]], axis=1)

    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.subplot()

    cols=np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    for cellind in srtd.index:
        stages = srtd.loc[cellind]
        dep = depths[cellind] + np.random.uniform(-15, 15) # add some jitter to avoid overlapping cells with same depth
        ax.plot(rates_znorm.loc[cellind, stages], [dep, dep], color=cols[stages[-1]], linewidth=0.75)
        ax.scatter(rates_znorm.loc[cellind, stages], [dep, dep], c=cols[rates_znorm.loc[cellind, stages].index])
    ax.set_xlabel('Firing rate (z-score)')
    ax.set_ylabel('Depths, um')

    import matplotlib.patches as mpatches
    patches = []
    for col, name in zip(cols, stage_names):
        patches.append(mpatches.Patch(color=col, label=name))
        # red_patch = mpatches.Patch(color='red', label='The red data')
    plt.legend(handles=patches)

    ax.invert_yaxis()
    plt.show()


if __name__ == "__main__":
    plot_cells_rates(*load([8, 12], 'm1_m1'))

    # bands = [[4,8], [8, 12], [12, 25], [25, 35], [45, 55]]

    start = 2
    step = 4.5
    windwidth = 4

    bands = [[start+i*step, start+i*step + windwidth] for i in range(24)]
    # bands.append([74.0, 78.0])
    bands.insert(1, [4, 8])
    bands.insert(3, [8, 12])

    label = 'detailed/m1_m1'
    for band in [[4,8], [8,12]]:
        data_per_period, titles = load(band, files_label=label)
        plot_per_band(data_per_period, titles, band, fig_title='ppc')

    plot_per_band = False
    fig, axs, grand_max_r, grand_min_r = plot_all_avg(bands, MOI_PPC, plot_per_band, files_label=label, save_fig_title=f'm1_m1_ppc_heat')


    label = 'detailed/joyst_x_m1_m1'
    fig, axs,  grand_max_r, grand_min_r = plot_all_avg(bands, MOI_PPC, plot_per_band, files_label=label, color='tab:olive') # , figAndAxs=(fig, axs), grand_max_r=grand_max_r, grand_min_r=grand_min_r

    label = 'detailed/joyst_y_m1_m1'
    plot_all_avg(bands, MOI_PPC, plot_per_band, files_label=label, save_fig_title=None, color='tab:olive', figAndAxs=(fig, axs), grand_max_r=grand_max_r, grand_min_r=grand_min_r) # f'm1_m1_joyst_ppc'

    label = 'detailed/joyst_x_m1_m1_permut'
    plot_all_avg(bands, MOI_PPC, plot_per_band, files_label=label, save_fig_title=None, color='tab:red', figAndAxs=(fig, axs), grand_max_r=grand_max_r, grand_min_r=grand_min_r) # f'm1_m1_joyst_ppc'

    label = 'detailed/joyst_y_m1_m1_permut'
    plot_all_avg(bands, MOI_PPC, plot_per_band, files_label=label, save_fig_title='m1_joyst_ppc', color='tab:red', figAndAxs=(fig, axs), grand_max_r=grand_max_r, grand_min_r=grand_min_r) # f'm1_m1_joyst_ppc'


