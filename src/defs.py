"""
defs.py

Definition of the cells and auxiliar functions used in the model

Contributors: romanbaravalle@gmail.com
"""
from netpyne import specs
import gc

#------------------------------------------------------------------------------
## Function to calculate the fitness according to required rate
def rateFitnessFunc(simData, print=False, **kwargs):
    import numpy as np
    pops = kwargs['pops']
    maxFitness = kwargs['maxFitness']

    popFitness = [min(np.exp(abs(v['target'] - simData['popRates'][k]) / v['width']), maxFitness)
                  if simData['popRates'][k] > v['min'] else maxFitness for k, v in pops.items()]
    fitness = np.mean(popFitness)

    if print:
        popInfo = '; '.join(
            ['%s rate=%.1f fit=%1.f' % (p, simData['popRates'][p], popFitness[i]) for i, p in enumerate(pops)])
        print('  ' + popInfo)
    return fitness

#------------------------------------------------------------------------------
## Function to modify cell params during sim (e.g. modify PT ih)
def modifyMechsFunc(simTime, cfg):
    from netpyne import sim

    t = simTime

    cellType = cfg.modifyMechs['cellType']
    mech = cfg.modifyMechs['mech']
    prop = cfg.modifyMechs['property']
    newFactor = cfg.modifyMechs['newFactor']
    origFactor = cfg.modifyMechs['origFactor']
    factor = newFactor / origFactor
    change = False

    if cfg.modifyMechs['endTime']-1.0 <= t <= cfg.modifyMechs['endTime']+1.0:
        factor = origFactor / newFactor if abs(newFactor) > 0.0 else origFactor
        change = True

    elif t >= cfg.modifyMechs['startTime']-1.0 <= t <= cfg.modifyMechs['startTime']+1.0:
        factor = newFactor / origFactor if abs(origFactor) > 0.0 else newFactor
        change = True

    if change:
        print('   Modifying %s %s %s by a factor of %f' % (cellType, mech, prop, factor))
        for cell in sim.net.cells:
            if 'cellType' in cell.tags and cell.tags['cellType'] == cellType:
                for secName, sec in cell.secs.items():
                    if mech in sec['mechs'] and prop in sec['mechs'][mech]:
                        # modify python
                        sec['mechs'][mech][prop] = [g * factor for g in sec['mechs'][mech][prop]] if isinstance(sec['mechs'][mech][prop], list) else sec['mechs'][mech][prop] * factor

                        # modify neuron
                        for iseg, seg in enumerate(sec['hObj']):  # set mech params for each segment
                            if sim.cfg.verbose: print('   Modifying %s %s %s by a factor of %f' % (secName, mech, prop, factor))
                            setattr(getattr(seg, mech), prop, getattr(getattr(seg, mech), prop) * factor)
    return None

def reducedCellModels(label, p, cwd, layer, cfg, reducedSecList, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label=label, conds={'cellType': label[0:2], 'cellModel': 'HH_reduced', 'ynorm': layer[p['layer']]},
    fileName=cwd+'/cells/'+p['cname']+'.py', cellName=p['cname'], cellArgs={'params': p['carg']} if p['carg'] else None)
    dendL = (layer[p['layer']][0]+(layer[p['layer']][1]-layer[p['layer']][0])/2.0) * cfg.sizeY  # adapt dend L based on layer
    for secName in ['Adend1', 'Adend2', 'Adend3', 'Bdend']: cellRule['secs'][secName]['geom']['L'] = dendL / 3.0  # update dend L
    for k,v in reducedSecList.items(): cellRule['secLists'][k] = v  # add secLists
    netParamsAux.addCellParamsWeightNorm(label, cwd+'/conn/'+label+'_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # add weightNorm

    # set 3d points
    offset, prevL = 0, 0
    somaL = netParamsAux.cellParams[label]['secs']['soma']['geom']['L']
    for secName, sec in netParamsAux.cellParams[label]['secs'].items():
        sec['geom']['pt3d'] = []
        if secName in ['soma', 'Adend1', 'Adend2', 'Adend3']:  # set 3d geom of soma and Adends
            sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
            prevL = float(prevL + sec['geom']['L'])
            sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
        if secName in ['Bdend']:  # set 3d geom of Bdend
            sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
            sec['geom']['pt3d'].append([offset+sec['geom']['L'], somaL, 0, sec['geom']['diam']])        
        if secName in ['axon']:  # set 3d geom of axon
            sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
            sec['geom']['pt3d'].append([offset+0, -sec['geom']['L'], 0, sec['geom']['diam']])   

    if saveCellParams: netParamsAux.saveCellParamsRule(label=label, fileName=cwd+'/cells/'+label+'_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def PT5BFullModel(cfg, cwd, saveCellParams):
    netParamsAux = specs.NetParams()
    ihMod2str = {'harnett': 1, 'kole': 2, 'migliore': 3}
    cellRule = netParamsAux.importCellParams(label='PT5B_full', conds={'cellType': 'PT', 'cellModel': 'HH_full'},
      fileName=cwd+'/cells/PTcell.hoc', cellName='PTcell', cellArgs=[ihMod2str[cfg.ihModel], cfg.ihSlope], somaAtOrigin=True)
    nonSpiny = ['apic_0', 'apic_1']
    netParamsAux.addCellParamsSecList(label='PT5B_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    netParamsAux.addCellParamsSecList(label='PT5B_full', secListName='below_soma', somaDistY=[-600, 0])  # sections within 0-300 um of soma
    for sec in nonSpiny: cellRule['secLists']['perisom'].remove(sec)
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]
    # Adapt ih params based on cfg param
    for secName in cellRule['secs']:
        for mechName,mech in cellRule['secs'][secName]['mechs'].items():
            if mechName in ['ih','h','h15', 'hd']: 
                mech['gbar'] = [g*cfg.ihGbar for g in mech['gbar']] if isinstance(mech['gbar'],list) else mech['gbar']*cfg.ihGbar
                if cfg.ihModel == 'migliore':   
                    mech['clk'] = cfg.ihlkc  # migliore's shunt current factor
                    mech['elk'] = cfg.ihlke  # migliore's shunt current reversal potential
                if secName.startswith('dend'): 
                    mech['gbar'] *= cfg.ihGbarBasal  # modify ih conductance in soma+basal dendrites
                    mech['clk'] *= cfg.ihlkcBasal  # modify ih conductance in soma+basal dendrites
                if secName in cellRule['secLists']['below_soma']: #secName.startswith('dend'): 
                    mech['clk'] *= cfg.ihlkcBelowSoma  # modify ih conductance in soma+basal dendrites
    # Reduce dend Na to avoid dend spikes (compensate properties by modifying axon params)
    for secName in cellRule['secLists']['alldend']:
        cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0153130368342 * cfg.dendNa # 0.25 
    cellRule['secs']['soma']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.somaNa
    cellRule['secs']['axon']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.axonNa # 11  
    cellRule['secs']['axon']['geom']['Ra'] = 137.494564931 * cfg.axonRa # 0.005
    # Remove Na (TTX)
    if cfg.removeNa:
        for secName in cellRule['secs']: cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0
    netParamsAux.addCellParamsWeightNorm('PT5B_full', cwd+'/conn/PT5B_full_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # load weight norm
    if saveCellParams: netParamsAux.saveCellParamsRule(label='PT5B_full', fileName=cwd+'/cells/PT5B_full_cellParams.pkl')

    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def IT5AFullModel(cwd, saveCellParams, cfg, layer):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='IT5A_full', conds={'cellType': 'IT', 'cellModel': 'HH_full', 'ynorm': layer['5A']},
    fileName=cwd+'/cells/ITcell.py', cellName='ITcell', cellArgs={'params': 'BS1579'}, somaAtOrigin=True)
    netParamsAux.renameCellParamsSec(label='IT5A_full', oldSec='soma_0', newSec='soma')
    netParamsAux.addCellParamsWeightNorm('IT5A_full', cwd+'/conn/IT_full_BS1579_weightNorm.pkl', threshold=cfg.weightNormThreshold) # add weightNorm before renaming soma_0
    netParamsAux.addCellParamsSecList(label='IT5A_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # basal+apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in ['apic_0', 'apic_1']]
    if saveCellParams: netParamsAux.saveCellParamsRule(label='IT5A_full', fileName=cwd+'/cells/IT5A_full_cellParams.pkl')
    
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def IT5BFullModel(cwd, layer, saveCellParams): # NOT USED
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='IT5B_full', conds={'cellType': 'IT', 'cellModel': 'HH_full', 'ynorm': layer['5B']},
    fileName=cwd+'/cells/ITcell.py', cellName='ITcell', cellArgs={'params': 'BS1579'}, somaAtOrigin=True)
    netParamsAux.addCellParamsSecList(label='IT5B_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # basal+apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in ['apic_0', 'apic_1']]
    netParamsAux.addCellParamsWeightNorm('IT5B_full', cwd+'/conn/IT_full_BS1579_weightNorm.pkl')
    if saveCellParams: netParamsAux.saveCellParamsRule(label='IT5B_full', fileName=cwd+'/cells/IT5B_full_cellParams.pkl')

    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def PVReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='PV_reduced', conds={'cellType':'PV', 'cellModel':'HH_reduced'}, 
    fileName=cwd+'/cells/FS3.hoc', cellName='FScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    netParamsAux.addCellParamsWeightNorm('PV_reduced', cwd+'/conn/PV_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    # cellRule['secs']['soma']['weightNorm'][0] *= 1.5
    if saveCellParams: netParamsAux.saveCellParamsRule(label='PV_reduced', fileName=cwd+'/cells/PV_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def SOMReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='SOM_reduced', conds={'cellType':'SOM', 'cellModel':'HH_reduced'}, 
    fileName=cwd+'/cells/LTS3.hoc', cellName='LTScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    netParamsAux.addCellParamsWeightNorm('SOM_reduced', cwd+'/conn/SOM_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParamsAux.saveCellParamsRule(label='SOM_reduced', fileName=cwd+'/cells/SOM_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def VIPReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='VIP_reduced', conds={'cellType': 'VIP', 'cellModel': 'HH_reduced'},
                                            fileName=cwd+'/cells/vipcr_cell.hoc',         
                                            cellName='VIPCRCell_EDITED', importSynMechs = True)
    cellRule['secLists']['spiny'] = ['soma', 'rad1', 'rad2', 'ori1', 'ori2']
    netParamsAux.addCellParamsWeightNorm('VIP_reduced', cwd+'/conn/VIP_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParamsAux.saveCellParamsRule(label='VIP_reduced', fileName=cwd+'/cells/VIP_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def NGFReducedModel(cwd, cfg, saveCellParams):
    netParamsAux = specs.NetParams()
    cellRule = netParamsAux.importCellParams(label='NGF_reduced', conds={'cellType': 'NGF', 'cellModel': 'HH_reduced'}, 
                                        fileName=cwd+'/cells/ngf_cell.hoc',
                                        cellName='ngfcell', importSynMechs = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    netParamsAux.addCellParamsWeightNorm('NGF_reduced', cwd+'/conn/NGF_reduced_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    cellRule['secs']['soma']['weightNorm'][0] *= 1.5
    cellRule['secs']['soma']['weightNorm'][0] *= 1.5
    if saveCellParams: netParamsAux.saveCellParamsRule(label='NGF_reduced', fileName=cwd+'/cells/NGF_reduced_cellParams.pkl')
    del netParamsAux
    gc.collect()  # collect garbage to free memory
    # return the cell rule
    return cellRule

def definePops(netParams, cfg, layer, density):
    ## Local populations
    ### Layer 1:
    netParams.popParams['NGF1']  =   {'cellModel': 'HH_reduced', 'cellType': 'NGF', 'ynormRange': layer['1'], 'density': density[('M1','nonVIP')][0]}

    ### Layer 2/3:
    netParams.popParams['IT2']  =   {'cellModel': cfg.cellmod['IT2'],  'cellType': 'IT', 'ynormRange': layer['2'], 'density': density[('M1','E')][1]}
    netParams.popParams['SOM2'] =   {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['2'], 'density': density[('M1','SOM')][1]}
    netParams.popParams['PV2']  =   {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['2'], 'density': density[('M1','PV')][1]}
    netParams.popParams['VIP2']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['2'], 'density': density[('M1','VIP')][1]}
    netParams.popParams['NGF2']  =  {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['2'], 'density': density[('M1','nonVIP')][1]}

    ### Layer 4:
    netParams.popParams['IT4']  =   {'cellModel': cfg.cellmod['IT4'],  'cellType': 'IT', 'ynormRange': layer['4'], 'density': density[('M1','E')][2]}
    netParams.popParams['SOM4'] =   {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['4'], 'density': density[('M1','SOM')][2]}
    netParams.popParams['PV4']  =   {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['4'], 'density': density[('M1','PV')][2]}
    netParams.popParams['VIP4']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['4'], 'density': density[('M1','VIP')][2]}
    netParams.popParams['NGF4']  =  {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['4'], 'density': density[('M1','nonVIP')][2]}

    ### Layer 5A:
    netParams.popParams['IT5A'] =   {'cellModel': cfg.cellmod['IT5A'], 'cellType': 'IT', 'ynormRange': layer['5A'], 'density': density[('M1','E')][3]}
    netParams.popParams['SOM5A'] =  {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['5A'], 'density': density[('M1','SOM')][3]}
    netParams.popParams['PV5A']  =  {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['5A'], 'density': density[('M1','PV')][3]}
    netParams.popParams['VIP5A']  = {'cellModel': 'HH_reduced',         'cellType': 'VIP', 'ynormRange': layer['5A'], 'density': density[('M1','VIP')][3]}
    netParams.popParams['NGF5A']  = {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['5A'], 'density': density[('M1','nonVIP')][3]}

    ### Layer 5B:
    netParams.popParams['IT5B'] =   {'cellModel': cfg.cellmod['IT5B'], 'cellType': 'IT', 'ynormRange': layer['5B'], 'density': 0.5*density[('M1','E')][4]}
    netParams.popParams['PT5B'] =   {'cellModel': cfg.cellmod['PT5B'], 'cellType': 'PT', 'ynormRange': layer['5B'], 'density': 0.5*density[('M1','E')][4]}
    netParams.popParams['SOM5B'] =  {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['5B'], 'density': density[('M1','SOM')][4]}
    netParams.popParams['PV5B']  =  {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['5B'], 'density': density[('M1','PV')][4]}
    netParams.popParams['VIP5B']  = {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['5B'], 'density': density[('M1','VIP')][4]}
    netParams.popParams['NGF5B']  = {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['5B'], 'density': density[('M1','nonVIP')][4]}

    ### Layer 6:
    netParams.popParams['IT6']  =   {'cellModel': cfg.cellmod['IT6'],  'cellType': 'IT', 'ynormRange': layer['6'],  'density': 0.5*density[('M1','E')][5]}
    netParams.popParams['CT6']  =   {'cellModel': cfg.cellmod['CT6'],  'cellType': 'CT', 'ynormRange': layer['6'],  'density': 0.5*density[('M1','E')][5]}
    netParams.popParams['SOM6'] =   {'cellModel': 'HH_reduced',         'cellType': 'SOM','ynormRange': layer['6'],  'density': density[('M1','SOM')][5]}
    netParams.popParams['PV6']  =   {'cellModel': 'HH_reduced',         'cellType': 'PV', 'ynormRange': layer['6'],  'density': density[('M1','PV')][5]}
    netParams.popParams['VIP6']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['6'], 'density': density[('M1','VIP')][1]}
    netParams.popParams['NGF6']  =  {'cellModel': 'HH_reduced',         'cellType': 'NGF', 'ynormRange': layer['6'], 'density': density[('M1','nonVIP')][1]}

    return None

def addLongConnections(cwd, netParams, cfg, layer):
    #TODO: Check and rewrite to load in-vivo spikes
    import pickle, json
    ## load experimentally based parameters for long range inputs
    with open(cwd + '/conn/conn_long.pkl', 'rb') as fileObj:
        connLongData = pickle.load(fileObj)
    # ratesLong = connLongData['rates']

    numCells = cfg.numCellsLong
    noise = cfg.noiseLong
    start = cfg.startLong

    if cfg.addInVivoThalamus: 
        longPops = ['TPO', 'S1', 'S2', 'cM1', 'M2', 'OC']
    else:
        longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']
    ## create populations with fixed
    for longPop in longPops:
        netParams.popParams[longPop] = {'cellModel': 'VecStim', 'numCells': numCells, 'rate': cfg.ratesLong[longPop],
                                        'noise': noise, 'start': start, 'pulses': [],
                                        'ynormRange': layer['long' + longPop]}
        if isinstance(cfg.ratesLong[longPop], str):  # filename to load spikes from
            spikesFile = cfg.ratesLong[longPop]
            with open(spikesFile, 'r') as f: spks = json.load(f)
            netParams.popParams[longPop].pop('rate')
            netParams.popParams[longPop]['spkTimes'] = spks

    if cfg.addInVivoThalamus:
        netParams.popParams['TVL'] = {'cellModel': 'VecStim',
                                                 'numCells': len(cfg.spikeTimesInVivo),
                                                 'spkTimes': cfg.spikeTimesInVivo,
                                                 'ynormRange': layer['long' + 'TVL']}
    return connLongData

def addStimPulses(cfg, netParams):
    for key in [k for k in dir(cfg) if k.startswith('pulse')]:
        params = getattr(cfg, key, None)
        [pop, start, end, rate, noise] = [params[s] for s in ['pop', 'start', 'end', 'rate', 'noise']]
        if 'duration' in params and params['duration'] is not None and params['duration'] > 0:
            end = start + params['duration']

        if pop in netParams.popParams:
            if 'pulses' not in netParams.popParams[pop]: netParams.popParams[pop]['pulses'] = {}    
            netParams.popParams[pop]['pulses'].append({'start': start, 'end': end, 'rate': rate, 'noise': noise})
    
    return None

def addStimIClamp(cfg, netParams):
    for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
        params = getattr(cfg, key, None)
        [pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]

        #cfg.analysis['plotTraces']['include'].append((pop,0))  # record that pop

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}
        
        # connect stim source to target
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop},
            'sec': sec, 
            'loc': loc}
    return None

def addStimNetStim(cfg, netParams, ESynMech, SOMESynMech):
    for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
        params = getattr(cfg, key, None)
        [pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
        [params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

        # cfg.analysis['plotTraces']['include'] = [(pop,0)]

        if synMech == ESynMech:
            wfrac = cfg.synWeightFractionEE
        elif synMech == SOMESynMech:
            wfrac = cfg.synWeightFractionSOME
        else:
            wfrac = [1.0] #TODO: What is the use of wfrac??

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

        # connect stim source to target
        # for i, syn in enumerate(synMech):
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop, 'ynorm': ynorm},
            'sec': sec, 
            'loc': loc,
            'synMech': synMech,
            'weight': weight,
            'synMechWeightFactor': synMechWeightFactor,
            'delay': delay}
    return None

def defineEEConnections(bins, cfg, netParams, cellModels, pmat, wmat):
    labelsConns = [('W+AS_norm', 'IT', 'L2/3,4'), ('W+AS_norm', 'IT', 'L5A,5B'), 
                   ('W+AS_norm', 'PT', 'L5B'), ('W+AS_norm', 'IT', 'L6'), ('W+AS_norm', 'CT', 'L6')]
    labelPostBins = [('W+AS', 'IT', 'L2/3,4'), ('W+AS', 'IT', 'L5A,5B'), ('W+AS', 'PT', 'L5B'), 
                    ('W+AS', 'IT', 'L6'), ('W+AS', 'CT', 'L6')]
    labelPreBins = ['W', 'AS', 'AS', 'W', 'W']
    preTypes = [['IT'], ['IT'], ['IT', 'PT'], ['IT','CT'], ['IT','CT']] 
    postTypes = ['IT', 'IT', 'PT', 'IT','CT']
    ESynMech = ['AMPA','NMDA']

    for i,(label, preBinLabel, postBinLabel) in enumerate(zip(labelsConns,labelPreBins, labelPostBins)):
        for ipre, preBin in enumerate(bins[preBinLabel]):
            for ipost, postBin in enumerate(bins[postBinLabel]):
                for cellModel in cellModels:
                    ruleLabel = 'EE_'+cellModel+'_'+str(i)+'_'+str(ipre)+'_'+str(ipost)
                    netParams.connParams[ruleLabel] = { 
                        'preConds': {'cellType': preTypes[i], 'ynorm': list(preBin)}, 
                        'postConds': {'cellModel': cellModel, 'cellType': postTypes[i], 'ynorm': list(postBin)},
                        'synMech': ESynMech,
                        'probability': pmat[label][ipost,ipre],
                        'weight': wmat[label][ipost,ipre] * cfg.EEGain / cfg.synsperconn[cellModel], 
                        'synMechWeightFactor': cfg.synWeightFractionEE,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': cfg.synsperconn[cellModel],
                        'sec': 'spiny'}

    return None

def defineEIConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, wmat):
    binsLabel = 'inh'
    preTypes = excTypes
    postTypes = inhTypes
    ESynMech = ['AMPA','NMDA']
    for i,postType in enumerate(postTypes):
        for ipre, preBin in enumerate(bins[binsLabel]):
            for ipost, postBin in enumerate(bins[binsLabel]):
                ruleLabel = 'EI_'+str(i)+'_'+str(ipre)+'_'+str(ipost)+'_'+str(postType)
                netParams.connParams[ruleLabel] = {
                    'preConds': {'cellType': preTypes, 'ynorm': list(preBin)},
                    'postConds': {'cellType': postType, 'ynorm': list(postBin)},
                    'synMech': ESynMech,
                    'probability': pmat[('E', postType)][ipost,ipre],
                    'weight': wmat[('E', postType)][ipost,ipre] * cfg.EIGain * cfg.EICellTypeGain[postType],
                    'synMechWeightFactor': cfg.synWeightFractionEI,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'sec': 'soma'} # simple I cells used right now only have soma
    return None

def defineIEConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech):
    binsLabel = 'inh'
    preTypes = inhTypes
    synMechs = [PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech] 
    weightFactors = [[1.0], cfg.synWeightFractionSOME, [1.0], cfg.synWeightFractionNGF] # Update VIP and NGF syns! 
    secs = ['perisom', 'apicdend', 'apicdend', 'apicdend']
    postTypes = excTypes
    for ipreType, (preType, synMech, weightFactor, sec) in enumerate(zip(preTypes, synMechs, weightFactors, secs)):
        for ipostType, postType in enumerate(postTypes):
            for ipreBin, preBin in enumerate(bins[binsLabel]):
                for ipostBin, postBin in enumerate(bins[binsLabel]):
                    for cellModel in ['HH_reduced', 'HH_full']:
                        ruleLabel = preType+'_'+postType+'_'+cellModel+'_'+str(ipreBin)+'_'+str(ipostBin)
                        netParams.connParams[ruleLabel] = {
                            'preConds': {'cellType': preType, 'ynorm': list(preBin)},
                            'postConds': {'cellModel': cellModel, 'cellType': postType, 'ynorm': list(postBin)},
                            'synMech': synMech,
                            'probability': '%f * exp(-dist_3D_border/probLambda)' % (pmat[(preType, 'E')][ipostBin,ipreBin]),
                            'weight': cfg.IEweights[ipostBin] * cfg.IEGain/ cfg.synsperconn[cellModel],
                            'synMechWeightFactor': weightFactor,
                            'synsPerConn': cfg.synsperconn[cellModel],
                            'delay': 'defaultDelay+dist_3D/propVelocity',
                            'sec': sec} # simple I cells used right now only have soma
    return None

def defineIIConnections(excTypes, inhTypes, bins, cfg, netParams, pmat, PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech):
    binsLabel = 'inh'
    preTypes = inhTypes
    synMechs =  [PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech]   
    sec = 'perisom'
    postTypes = inhTypes
    for ipre, (preType, synMech) in enumerate(zip(preTypes, synMechs)):
        for ipost, postType in enumerate(postTypes):
            for iBin, bin in enumerate(bins[binsLabel]):
                for cellModel in ['HH_reduced']:
                    ruleLabel = preType+'_'+postType+'_'+str(iBin)
                    netParams.connParams[ruleLabel] = {
                        'preConds': {'cellType': preType, 'ynorm': bin},
                        'postConds': {'cellModel': cellModel, 'cellType': postType, 'ynorm': bin},
                        'synMech': synMech,
                        'probability': '%f * exp(-dist_3D_border/probLambda)' % (pmat[(preType, postType)]),
                        'weight': cfg.IIweights[iBin] * cfg.IIGain / cfg.synsperconn[cellModel],
                        'synsPerConn': cfg.synsperconn[cellModel],
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'sec': sec} # simple I cells used right now only have soma
    return None

def defineLongRangeConnections(connLongData, cfg, netParams, cellModels, ESynMech):
    # load load experimentally based parameters for long range inputs
    cmatLong = connLongData['cmat']
    binsLong = connLongData['bins']

    longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']
    cellTypes = ['IT', 'PT', 'CT', 'PV', 'SOM', 'VIP', 'NGF']
    EorI = ['exc', 'inh']
    syns = {'exc': ESynMech, 'inh': 'GABAA'}
    synFracs = {'exc': cfg.synWeightFractionEE, 'inh': [1.0]}

    for longPop in longPops:
        for ct in cellTypes:
            for EorI in ['exc', 'inh']:
                for i, (binRange, convergence) in enumerate(zip(binsLong[(longPop, ct)], cmatLong[(longPop, ct, EorI)])):
                    for cellModel in cellModels:
                        ruleLabel = longPop+'_'+ct+'_'+EorI+'_'+cellModel+'_'+str(i)
                        netParams.connParams[ruleLabel] = { 
                            'preConds': {'pop': longPop}, 
                            'postConds': {'cellModel': cellModel, 'cellType': ct, 'ynorm': list(binRange)},
                            'synMech': syns[EorI],
                            'convergence': convergence,
                            'weight': cfg.weightLong[longPop] / cfg.synsperconn[cellModel], 
                            'synMechWeightFactor': cfg.synWeightFractionEE,
                            'delay': 'defaultDelay+dist_3D/propVelocity',
                            'synsPerConn': cfg.synsperconn[cellModel],
                            'sec': 'spiny'}
    return None

def defineSubcellularConnectivity(cwd, netParams, layer, ESynMech, SOMESynMech, VIPSynMech, NGFSynMech, inhTypes):
    import json
    with open(cwd+'/conn/conn_dend_PT.json', 'r') as fileObj: connDendPTData = json.load(fileObj)
    with open(cwd+'/conn/conn_dend_IT.json', 'r') as fileObj: connDendITData = json.load(fileObj)
    
    #------------------------------------------------------------------------------
    # L2/3,TVL,S2,cM1,M2 -> PT (Suter, 2015)
    lenY = 30 
    spacing = 50
    gridY = list(range(0, -spacing*lenY, -spacing))
    synDens, _, fixedSomaY = connDendPTData['synDens'], connDendPTData['gridY'], connDendPTData['fixedSomaY']
    for k in synDens.keys():
        prePop,postType = k.split('_')  # eg. split 'M2_PT'
        if prePop == 'L2': prePop = 'IT2'  # include conns from layer 2/3 and 4
        netParams.subConnParams[k] = {
        'preConds': {'pop': prePop}, 
        'postConds': {'cellType': postType},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '1Dmap', 'gridX': None, 'gridY': gridY, 'gridValues': synDens[k], 'fixedSomaY': fixedSomaY}} 

    #------------------------------------------------------------------------------
    # TPO, TVL, M2, OC  -> E (L2/3, L5A, L5B, L6) (Hooks 2013)
    lenY = 26
    spacing = 50
    gridY = list(range(0, -spacing*lenY, -spacing))
    synDens, _, fixedSomaY = connDendITData['synDens'], connDendITData['gridY'], connDendITData['fixedSomaY']
    for k in synDens.keys():
        prePop,post = k.split('_')  # eg. split 'M2_L2'
        postCellTypes = ['IT','PT','CT'] if prePop in ['OC','TPO'] else ['IT','CT']  # only OC,TPO include PT cells
        postyRange = list(layer[post.split('L')[1]]) # get layer yfrac range 
        if post == 'L2': postyRange[1] = layer['4'][1]  # apply L2 rule also to L4 
        netParams.subConnParams[k] = {
        'preConds': {'pop': prePop}, 
        'postConds': {'ynorm': postyRange , 'cellType': postCellTypes},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '1Dmap', 'gridX': None, 'gridY': gridY, 'gridValues': synDens[k], 'fixedSomaY': fixedSomaY}} 

    #------------------------------------------------------------------------------
    # S1, S2, cM1 -> E IT/CT; no data, assume uniform over spiny
    netParams.subConnParams['S1,S2,cM1->IT,CT'] = {
        'preConds': {'pop': ['S1','S2','cM1']}, 
        'postConds': {'cellType': ['IT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # rest of local E->E (exclude IT2->PT); uniform distribution over spiny
    netParams.subConnParams['IT2->non-PT'] = {
        'preConds': {'pop': ['IT2']}, 
        'postConds': {'cellType': ['IT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 
        
    netParams.subConnParams['non-IT2->E'] = {
        'preConds': {'pop': ['IT4','IT5A','IT5B','PT5B','IT6','CT6']}, 
        'postConds': {'cellType': ['IT','PT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # PV->E; perisomatic (no sCRACM)
    netParams.subConnParams['PV->E'] = {
        'preConds': {'cellType': 'PV'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'perisom', 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # SOM->E; apical dendrites (no sCRACM)
    netParams.subConnParams['SOM->E'] = {
        'preConds': {'cellType': 'SOM'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'apicdend',
        'groupSynMechs': SOMESynMech,
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # VIP->E; apical dendrites (no sCRACM)
    netParams.subConnParams['VIP->E'] = {
        'preConds': {'cellType': 'VIP'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'apicdend',
        'groupSynMechs': VIPSynMech,
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF->E; apical dendrites (no sCRACM)
    ## Add the following level of detail?
    # -- L1 NGF -> L2/3+L5 tuft
    # -- L2/3 NGF -> L2/3+L5 distal apical
    # -- L5 NGF -> L5 prox apical
    netParams.subConnParams['NGF->E'] = {
        'preConds': {'cellType': 'NGF'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'apicdend',
        'groupSynMechs': NGFSynMech,
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # All->I; apical dendrites (no sCRACM)
    netParams.subConnParams['All->I'] = {
        'preConds': {'cellType': ['IT', 'CT', 'PT'] + inhTypes},# + longPops}, 
        'postConds': {'cellType': inhTypes},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech,
        'density': 'uniform'} 

    return None

def loadInVivoSpikes():
    
    return None 





