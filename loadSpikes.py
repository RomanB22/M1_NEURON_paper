import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_epoched_spikes(path, region):
    spikes = pd.read_csv(path / f'{region}_epoched_spikes.csv')
    with open(path / f'{region}_epoched_spikes_attrs.json', 'r') as jsonfile:
	    attrs = json.load(jsonfile)

    for key, value in attrs.items():
        if isinstance(value, list):
            spikes.attrs[key] = np.array(value)
        else:
            spikes.attrs[key] = value

    return spikes

def loadInVivoSpikes(PathString, cwd, cfg, Trial = 0):
    m1_spikes = load_epoched_spikes(Path(PathString), 'm1')
    norm_sampled_depths = m1_spikes.attrs['cell_depths']/cfg.sizeY
    norm_sampled_depths[norm_sampled_depths>=1] = 0.99

    numSampledCellsPerLayer = [len(norm_sampled_depths[(norm_sampled_depths>=cfg.normLayers[i][0])
								& (norm_sampled_depths<cfg.normLayers[i][1])]) for i in cfg.normLayers.keys()]

    thalamus_spikes = load_epoched_spikes(Path(cwd+'/data/spikingData'), 'th')

    preToneTime = abs(thalamus_spikes.attrs['trial_window'][0])*1000
    postToneTime = abs(thalamus_spikes.attrs['trial_window'][1])*1000

	# cfg.transient = thalamus_spikes.attrs['margin']*1000
	# cfg.preTone = abs(thalamus_spikes.attrs['trial_window'][0])*1000-cfg.transient
	# cfg.postTone = thalamus_spikes.attrs['trial_window'][1]*1000-cfg.transient
	# cfg.duration = 2*cfg.transient + cfg.preTone + cfg.postTone

    cells = np.unique(thalamus_spikes['cell_id'])
    maskTrial = np.array(thalamus_spikes['trial'])==Trial
    maskBeforeUnlock = np.array(thalamus_spikes['stage'])==0
    maskUnlock = np.array(thalamus_spikes['stage'])==1
    maskToneOn = np.array(thalamus_spikes['stage'])==2
    maskToneOff = np.array(thalamus_spikes['stage'])==3

    thalamus_spikesBeforeUnlock = thalamus_spikes[maskTrial*maskBeforeUnlock]
    thalamus_spikesmaskUnlock = thalamus_spikes[maskTrial*maskUnlock]
    thalamus_spikesmaskToneOn = thalamus_spikes[maskTrial*maskToneOn]
    thalamus_spikesmaskToneOff = thalamus_spikes[maskTrial*maskToneOff]

    MarginTime = thalamus_spikesBeforeUnlock.iloc[0]['spike_time']
    UnlockTime = thalamus_spikesmaskUnlock.iloc[0]['spike_time']
    ToneOnTime = thalamus_spikesmaskToneOn.iloc[0]['spike_time']
    ToneOffTime = thalamus_spikesmaskToneOff.iloc[0]['spike_time']

    spikeTimesInVivo = [thalamus_spikes[maskTrial * (np.array(thalamus_spikes['cell_id'])==i)]['spike_time'].values.tolist()-UnlockTime for i in cells]

    # cfg.preTone, cfg.postTone

    for idx in range(len(spikeTimesInVivo)):
        spikes = spikeTimesInVivo[idx]*1000.
        spikeTimesInVivo[idx] = list(spikes[(spikes>=-cfg.preTone)*(spikes<=cfg.postTone)]+cfg.preTone)

    return spikeTimesInVivo, numSampledCellsPerLayer