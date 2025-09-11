"""
defs.py

Definition of the cells and auxiliar functions used in the model

Contributors: romanbaravalle@gmail.com
"""
from netpyne import specs
import gc
import random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union
import math

#------------------------------------------------------------------------------
## Function to calculate the fitness according to required rate
def rateFitnessFunc(simData, extraConds=False, **kwargs):
    import numpy as np
    pops = kwargs['pops']
    maxFitness = kwargs['maxFitness']

    factor=1
    # Add extra conditions to the fitness. It 'breaks' the fitness function
    if extraConds:
        # check I > E in each layer
        condsIE_L23 = (simData['popRates']['PV2'] > simData['popRates']['IT2']) and (simData['popRates']['SOM2'] > simData['popRates']['IT2'])
        condsIE_L5A = (simData['popRates']['PV5A'] > simData['popRates']['IT5A']) and (simData['popRates']['SOM5A'] > simData['popRates']['IT5A'])
        condsIE_L5B = (simData['popRates']['PV5B'] > simData['popRates']['IT5B']) and (simData['popRates']['SOM5B'] > simData['popRates']['IT5B'])
        condsIE_L6 = (simData['popRates']['PV6'] > simData['popRates']['IT6']) and (simData['popRates']['SOM6'] > simData['popRates']['IT6'])
        # check E L5 > L6 > L2
        condEE562_0 = (simData['popRates']['IT5A']+simData['popRates']['IT5B']+simData['popRates']['PT5B'])/3 > (simData['popRates']['IT6']+simData['popRates']['CT6'])/2
        condEE562_1 = (simData['popRates']['IT6']+simData['popRates']['CT6'])/2 > simData['popRates']['IT2']
        # check PV > SOM in each layer
        condsPVSOM_L23 = (simData['popRates']['PV2'] > simData['popRates']['SOM2'])
        condsPVSOM_L5A = (simData['popRates']['PV5A'] > simData['popRates']['SOM5A'])
        condsPVSOM_L5B = (simData['popRates']['PV5B'] > simData['popRates']['SOM5B'])
        condsPVSOM_L6 = (simData['popRates']['PV6'] > simData['popRates']['SOM6'])

        conds = [condsIE_L23, condsIE_L5A, condsIE_L5B, condsIE_L6, condEE562_0, condEE562_1, condsPVSOM_L23, condsPVSOM_L5A, condsPVSOM_L5B, condsPVSOM_L6]

        if not all(conds): factor = 1.5
        
    popFitness = [min(np.exp(factor*abs(v['target'] - simData['popRates'][k])/v['width']), maxFitness) 
                if simData['popRates'][k] > v['min'] else maxFitness for k,v in pops.items()]
    fitness = np.mean(popFitness)

    popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p, simData['popRates'][p], popFitness[i]) for i,p in enumerate(pops)])
    print('  '+popInfo)
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

    # weight = []

    for i,(label, preBinLabel, postBinLabel) in enumerate(zip(labelsConns,labelPreBins, labelPostBins)):
        for ipre, preBin in enumerate(bins[preBinLabel]):
            for ipost, postBin in enumerate(bins[postBinLabel]):
                for cellModel in cellModels:
                    ruleLabel = 'EE_'+cellModel+'_'+str(i)+'_'+str(ipre)+'_'+str(ipost)
                    # if 'PT' in  postTypes[i] and list(postBin)[0]>=0.47 and list(postBin)[1]<=0.8:
                    #     weight.append(wmat[label][ipost,ipre] * cfg.EEGain)
                    #     print(preTypes[i], postTypes[i], list(postBin), wmat[label][ipost,ipre] * cfg.EEGain)
                    #     print(min(weight), max(weight))
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
    # quit()
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

def SampleSpikes(spikeTimesList, cfg, preTone=-2., postTone=2, baselineEnd=-0.5, skipEmpty=False):
    # Check that the spiking input is enough to run the simulation
    if (cfg.SimulateBaseline==False and cfg.preTone>2000.):
        raise ValueError("cfg.preTone cannot be larger than 2000 ms") # TODO: Add extension for preTone: we could add more baseline to the left actually
    if (cfg.SimulateBaseline==False and cfg.postTone>2000.):
        raise ValueError("cfg.preTone cannot be larger than 2000 ms") # TODO: Add extension for postTone: could it be baseline again?

    MovementTrials = []
    BaselineTrials = []
    for spkList in spikeTimesList:
        MovementTrialsAux = []
        BaselineTrialsAux = []
        for spkTimes in spkList:
            if (preTone <= spkTimes <= baselineEnd): BaselineTrialsAux.append(1000*(spkTimes+abs(preTone)))
            if (-cfg.preTone/1000. <= spkTimes <= cfg.postTone/1000.): 
                PositiveTimes = 1000*spkTimes+cfg.preTone
                MovementTrialsAux.append(PositiveTimes)
        if skipEmpty:
            if len(MovementTrialsAux)>0: MovementTrials.append(MovementTrialsAux)
            if len(BaselineTrialsAux)>0: BaselineTrials.append(BaselineTrialsAux)
        else:
            MovementTrials.append(MovementTrialsAux)
            BaselineTrials.append(BaselineTrialsAux)
    # Sample spikes
    random.seed(cfg.seeds['tvl_sampling'])
    baselineSpks = random.choices(BaselineTrials, k=cfg.numCellsLong)
    baselineSpks = [list(i) for i in baselineSpks]

    movementAndPostSpks = random.choices(MovementTrials, k=cfg.numCellsLong)
    movementAndPostSpks = [list(i) for i in movementAndPostSpks]

    if cfg.SimulateBaseline==True:
        sampledSpikesSpan = 1000*(baselineEnd-preTone)
        numSpans = math.ceil(cfg.duration / sampledSpikesSpan)
        # TODO: Check the spike times extension (times should be unique and ordered)
        for num in range(numSpans-1):
            # Add a new sampling
            baselineSpksAux = random.choices(BaselineTrials, k=cfg.numCellsLong)
            baselineSpksAux = [[elem + (num+1)*sampledSpikesSpan for elem in sublist] for sublist in baselineSpksAux]
            baselineSpks = [a + b for a, b in zip(baselineSpks, baselineSpksAux)]

    return baselineSpks, movementAndPostSpks

def cellPerlayer(numbers):
    Layers = {'1': [0.0, 0.1*1350], '2': [0.1*1350,0.29*1350], '4': [0.29*1350,0.37*1350], '5A': [0.3*1350,0.47*1350], '5B': [0.47*1350,0.8*1350], '6': [0.8*1350, 1.0*1350]}

    from collections import defaultdict

    counts = defaultdict(int)

    for num in numbers:
        for layer, (low, high) in Layers.items():
            if low <= num < high:
                counts[layer] += 1
                break  # Assumes one number belongs to only one layer

    return counts

def loadThalSpikes(cwd, cfg, skipEmpty=False):
    import json
    with open(cwd+"/data/spikingData/ThRates.json", "r") as fileObj:
        data = json.loads(fileObj.read())

    spikeTimesList = []
    M1sampledCells = []
    foldersName = []

    for folder in data.keys():
        for i in range(len(data[folder].keys())-4): # exclude M1_cell_depths, Th_cell_depths, meanRate, stdRate
            spkid =  data[folder]['trial_%d' % i]['spkid']
            spkt = data[folder]['trial_%d' % i]['spkt']
            npre = int(np.max(spkid)) + 1
            spkTimes_by_cell = [[] for _ in range(npre)]
            for t, i in zip(spkt, spkid):
                spkTimes_by_cell[int(i)].append(float(t))
            spikeTimesList[len(spikeTimesList):] += spkTimes_by_cell
        cellDepths = data[folder]['M1_cell_depths']
        counts = cellPerlayer(cellDepths)
        M1sampledCells.append(counts)
        foldersName.append(folder)

    baselineSpks, movementAndPostSpks = SampleSpikes(spikeTimesList, cfg, skipEmpty=skipEmpty)

    return baselineSpks, movementAndPostSpks, M1sampledCells, foldersName

def average_dict_entries(dicts: List[Union[dict, defaultdict]]) -> Dict[str, float]:
    totals = defaultdict(int)
    counts = defaultdict(int)

    for entry in dicts:
        for key, value in entry.items():
            totals[key] += value
            counts[key] += 1

    averages = {key: int(totals[key] / counts[key]) for key in totals}
    return averages

def trimTVLSpikes(spikeList, cfg):
    trimmedList = []
    for i in spikeList:
        # We need to align the spike time to avoid numerical errors in the delivery of the vecStim (due torounding errors it could happen that the simulator find a negative delivery time, which stops the simulation)
        trimmedList.append(np.unique([round(np.round(j / cfg.dt) * cfg.dt, 2) for j in i if (0<j<cfg.duration)]).tolist())

    return trimmedList

def load_umap_results(reg='m1', n_components=2, period='scaled_prep'):
    import joblib
    
    filename = f'./manifolds/{period}/umap_results_n{n_components}_{reg}.pkl'
    loaded_results = joblib.load(filename)

    loaded_reprs = loaded_results['representations']
    loaded_reds = loaded_results['reductions']
    loaded_names = loaded_results['folder_names']
    task_progress = loaded_results['task_progress']
    validCellsDepth =  loaded_results['validCellsDepth']

    M1sampledCells = []
    RawData = []
    for i in range(len(loaded_names)):
        idx = np.argsort(validCellsDepth[i])
        counts = cellPerlayer(validCellsDepth[i][idx])
        M1sampledCells.append(counts)
        # RawData.append(loaded_reds[i]._raw_data[:,idx])
        RawData.append(loaded_reds[i][:,idx])
    import json
    params = json.load(open(f'./manifolds/UMAP_params.json', 'r'))

    return loaded_reprs, loaded_reds, loaded_names, task_progress, M1sampledCells, RawData, params

def overlapping_window(np_array, window_size=25):
    from scipy import ndimage
    return ndimage.uniform_filter1d(np_array, size=window_size, axis=1, mode='constant')

def non_overlapping_window(np_array, window_size=25):
    import math
    window_hop = window_size
    start_frame = window_size
    end_frame = window_hop * math.floor(float(np_array.shape[1]) / window_hop)
    window = []
    for frame_idx in range(start_frame, end_frame, window_hop):
        window.append(np.mean(np_array[:, frame_idx - window_size:frame_idx], axis=1))  # Add mean

    return np.transpose(np.vstack(window))

def sampleNeuronsFromModel(sim, cfg, plot=False):
    """
    Sample a given number of cells from each layer and plot in 3D.

    Parameters
    ----------
    sim : NetPyNE simulation object (with sim.net.cells populated)
    samples_to_pick : dict
        Keys are layer names (e.g., 'L1', 'L2/3', 'L4', 'L5', 'L6'),
        Values are number of cells to sample from each layer.

    Returns
    -------
    sampled_cells : dict
        Keys are layer names, values are list of sampled cell dicts
        with 'gid' and 'pos'.
    """
    # Sample neurons from the model following the same sampling of layers as in cfg.numSampledCellsPerLayer. Use same random seed to sample always the same neurons
    import random
    random.seed(cfg.seeds['m1_sampling'])
    
    # --- Helper to map yNorm -> layer ---
    def get_layer(y_norm):
        for layer, (ymin, ymax) in cfg.normLayers.items():
            if ymin <= y_norm < ymax:
                return layer
        return None
    
    # --- Collect cells by layer ---
    cells_by_layer = {layer: [] for layer in cfg.normLayers.keys()}
    for cell in sim.net.cells:
        y_norm = cell.tags.get('ynorm')
        if y_norm is None:
            continue
        layer = get_layer(y_norm)
        if layer:
            cells_by_layer[layer].append(cell.gid)

    # --- Sample from each layer ---
    sampled_cells = {}
    for layer, n in cfg.numSampledCellsPerLayer.items():
        if layer in cells_by_layer:
            sampled_cells[layer] = random.sample(
                cells_by_layer[layer],
                min(n, len(cells_by_layer[layer]))
            )
        else:
            sampled_cells[layer] = []

    if plot:
        import matplotlib.pyplot as plt
        # --- Plot sampled cells ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.tab10.colors  # distinct colors per layer
        for i, (layer, cells) in enumerate(sampled_cells.items()):
            xs = [c['pos'][0] for c in cells]
            ys = [c['pos'][1] for c in cells]
            zs = [c['pos'][2] for c in cells]
            ax.scatter(xs, ys, zs, label=f"Layer {layer}",
                    color=colors[i % len(colors)], s=40)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Sampled Cell Locations by Layer (using yNorm)")
        ax.legend()
        plt.show()

    return sampled_cells

def bin_spikes(spike_times, dt, wdw_start, wdw_end):
    # Function that puts spikes into bins
    edges = np.arange(wdw_start, wdw_end, dt)  # Get edges of time bins
    num_bins = edges.shape[0] - 1  # Number of bins
    num_neurons = spike_times.shape[0]  # Number of neurons
    neural_data = np.empty([num_bins, num_neurons])  # Initialize array for binned neural data
    # Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:, i] = np.histogram(spike_times[i], edges)[0]
    return neural_data

def binnedRaster(simData, cfg):
    fs = cfg.UMAP_params['fs']
    bin_time = cfg.UMAP_params['bin_time']
    spike_times = np.array(simData['spkt'].to_python())
    spike_ids = np.array(simData['spkid'].to_python())
    sampledCells = [j for i in cfg.sampled_cells.values() for j in i]
    sampledCells.sort()
    # print(sampledCells)
    spike_timesAux = []
    for i in sampledCells:
        spike_timesAux.append(spike_times[spike_ids == i]/1000.)
    spike_times = np.array(spike_timesAux, dtype=object)

    Raster = bin_spikes(spike_times, 1./fs, 0, cfg.duration/1000.+1./fs)
    Raster = overlapping_window(Raster, window_size=cfg.UMAP_params['window_size'])
    # Convert to rate
    Raster /= bin_time
    return Raster

def concatenateExpModelRate(ExpRaster, ModelRaster):
    # Concatenate the experimental and model rate data to calculate UMAP on the combined data
    import numpy as np
    ExpRaster= np.transpose(ExpRaster)
    ModelRaster= np.transpose(ModelRaster)
    Raster = np.hstack((ExpRaster, ModelRaster))
    ConcatenatedLabels = np.array([0]*np.shape(ExpRaster)[1] + [1]*np.shape(ModelRaster)[1])  # 0=ExpRaster, 1=ModelRaster
    # print(ConcatenatedLabels, np.shape(Raster))
    return Raster, ConcatenatedLabels

def UMAP(n_neighbors,min_dist,n_components,metric,randomNumber,Raster):
    import umap
    from scipy.stats import pearsonr
    umap_reduction = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                n_components=n_components,
                                metric=metric, random_state=randomNumber).fit(Raster.T)
    umap_representation = umap_reduction.transform(Raster.T)
    umap_representation_back = umap_reduction.inverse_transform(umap_representation).T
    (pearsonCorr, pvalue) = pearsonr(Raster.flatten(), umap_representation_back.flatten())

    return umap_representation, umap_reduction, pearsonCorr, pvalue

def calculateUMAP(Raster, cfg):
    n_neighbors, min_dist, metric, randomNumber = cfg.UMAP_params['n_neighbors'], cfg.UMAP_params['min_dist'], cfg.UMAP_params['metric'], cfg.UMAP_params['randomNumber']
    umap_representation, umap_reduction, pearsonCorr, pvalue = UMAP(n_neighbors, min_dist, n_components=cfg.n_components, metric=metric, randomNumber=randomNumber, Raster=Raster)
    return umap_representation, umap_reduction, pearsonCorr, pvalue

def umapFitnessFunc(umap_representation, ConcatenatedLabels):
    embeddingExp = umap_representation[ConcatenatedLabels==0, :]
    embeddingMod = umap_representation[ConcatenatedLabels==1, :]
    weightExp = np.ones((embeddingExp.shape[0],)) / embeddingExp.shape[0]
    weightMod = np.ones((embeddingMod.shape[0],)) / embeddingMod.shape[0]

    import ot  # pip install POT
    # Cost matrix = pairwise squared distances
    M = ot.dist(embeddingExp, embeddingMod, metric='euclidean')**2

    # Earth Mover’s Distance (Wasserstein-2 squared)
    emd2 = ot.emd2(weightExp, weightMod, M)
    wasserstein_dist = np.sqrt(emd2)

    sw_dist = ot.sliced.sliced_wasserstein_distance(embeddingExp, embeddingMod, n_projections=500)
    print("2D Wasserstein distance:", wasserstein_dist)
    print("Sliced Wasserstein distance:", sw_dist)

    return wasserstein_dist, sw_dist

def plot_embedding(embedding, labels, cfg, colors=("blue", "red"), alpha=0.7, size=50, title="Embedding"):
    """
    Plot 2D embedding with two subsets colored differently.
    
    Parameters
    ----------
    embedding : array-like, shape (n_samples, 2)
        The 2D embedding (e.g., UMAP output).
    labels : array-like, shape (n_samples,)
        Binary labels (0 or 1) indicating subset membership.
    colors : tuple
        Colors for the two subsets.
    alpha : float
        Transparency of points.
    size : int
        Point size.
    title : str
        Plot title.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    embedding = np.array(embedding)
    labels = np.array(labels)
    
    plt.figure(figsize=(8,6))
    plt.scatter(embedding[labels==0,0], embedding[labels==0,1],
                c=colors[0], alpha=alpha, s=size, label="Experimental")
    plt.scatter(embedding[labels==1,0], embedding[labels==1,1],
                c=colors[1], alpha=alpha, s=size, label="Model")
    plt.title(title)
    plt.legend()
    filename = cfg.saveFolder + "/" + cfg.simLabel + "_umap.png"
    plt.savefig(filename)
    plt.close()
