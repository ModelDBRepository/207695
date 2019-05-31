"""
(C) Asaph Zylbertal 01.10.16, HUJI, Jerusalem, Israel

Temporal integration protocol - two glomeruli with one shared cell, asynchronous stimulation (Article Fig 8)
If you use this model in your research please cite:
****************

"""

import sys
import glomeruli
import mitral_definition
import numpy as np
import neuron
import matplotlib.pyplot as plt


obP = 0


def ParallelSet(optt=0):
    '''#Set Neuron to run on threads and multisplit
    # optt =0 - start 1- off 2 - on
    '''
    global obP
    if optt == 0:
        neuron.h.load_file("parcom.hoc", "ParallelComputeTool")
        obP = neuron.h.ParallelComputeTool[0]
        obP.change_nthread(1, 1)
        obP.cacheeffic(1)
        obP.multisplit(1)
    if optt == 1:
        obP.cacheeffic(0)
        obP.multisplit(0)
    if optt == 2:
        obP.cacheeffic(1)
        obP.multisplit(1)


neuron.load_mechanisms('./channels')

to_plot = True
parts = 1000
simulate = True


params = {'dc': 0,
          'glu_delay': 1.0,
          'glu_syn_tau': 1000.0,
          'gaba_weight': 1.8e-5,
          'gaba_delay': 10.0,
          'gaba_syn_tau': 1000.0,
          'gaba_syn_e': -65,
          'glu_weight': 1.2e-5,
          'gap_g_base': 0.17,
          'noise_level': 0,
          'rm_factor': 1.1,
          'sim_time': 200000,
          'glu_on': True,
          'gaba_on': True,
          'morph_stdv': 0.5,
          'stim_amp': 0.04,
          'title': 'multi_glom_integration',
          'stimulated_cell': 'all'}


init_sim_time = 16000
stim_amp = 0.03
stim_delay = 10000
stim_dur = 4000

pc = neuron.h.ParallelContext()

# Init run on a single cell
mci = mitral_definition.full_mitral_neuron(
    rest_state_file='./rest_state',
     rm_factor=params['rm_factor'])

stim_stop = np.concatenate(
    [np.ones(stim_delay) * -(params['dc'] + 0.02),
     np.zeros(init_sim_time - (stim_delay))])
init_stim_vec = np.ones(init_sim_time) * params['dc'] + stim_stop
mci.init_vector_stim(
    np.linspace(0,
                init_sim_time,
                init_sim_time),
     init_stim_vec)

neuron.h.finitialize(-50.)
neuron.h.fcurrent()

neuron.run(init_sim_time)
vals = mci.save_states(None)

del mci

glom1 = glomeruli.Glomerulus(15, vals, params, pc, 0, rndseed=13)
glom2 = glomeruli.Glomerulus(13, vals, params, pc, 10000, rndseed=16)

glom1.connect(glom2, 0, 50000)

glom1.glom_stim(10000, 3500, params['stim_amp'])
glom2.glom_stim(95000, 3500, params['stim_amp'])

t = neuron.h.Vector()
t.record(neuron.h._ref_t)

v1 = neuron.h.Vector()
v1.record(glom1.mc[0].soma(0.5)._ref_v)
v2 = neuron.h.Vector()
v2.record(glom2.mc[0].soma(0.5)._ref_v)

pc.setup_transfer()

neuron.h.finitialize()
neuron.h.fcurrent()

part_len = params['sim_time'] / parts
if simulate:
    if to_plot:
        plt.ion()
        plt.figure()
        plt.hold(False)

    for part in range(parts):

        neuron.run((part + 1) * part_len)
        sys.stdout.write("\r%d / %d" % (part + 1, parts))
        if to_plot:
            plt.subplot(2, 1, 1)
            plt.plot(np.array(t), np.array(v1), 'b')
            plt.ylabel('Glom 1 cell Vm (mV)')
            plt.subplot(2, 1, 2)
            plt.plot(np.array(t), np.array(v2), 'g')
            plt.xlabel('t (ms)')
            plt.ylabel('Glom 2 cell Vm (mV)')
            plt.draw()
            plt.pause(0.0002)
        sys.stdout.flush()
