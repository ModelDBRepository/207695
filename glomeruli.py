"""
(C) Asaph Zylbertal 01.10.16, HUJI, Jerusalem, Israel

Connect multiple mitral cells to form a glomeruli, using excitatory, inhibitory and electrical synapses
If you use this model in your research please cite:
****************

"""

import mitral_definition
import neuron
import numpy as np


class Glomerulus():

    def __init__(self, cell_num, init_vals, params, pc, offset, rndseed=13):
        # create cells
        self.params = params
        self.cell_num = cell_num
        self.stim = []
        np.random.seed(rndseed)
        self.seeds = np.random.randint(0, 4294967294, cell_num)
        self.mc = [0] * cell_num

        for c in range(cell_num):

            self.mc[c] = mitral_definition.full_mitral_neuron(
                rest_vals=init_vals, num=c, rm_factor=params['rm_factor'],
                                               rel_morph_factor=params['morph_stdv'], rndseed=self.seeds[c])
        self.stim_vec = [0] * (cell_num)

        for c in range(cell_num):
            if params['noise_level'] > 0:
                self.stim_vec[c] = np.random.normal(
                    0,
                    params['noise_level'],
                    params['sim_time']) + np.ones(params['sim_time']) * params['dc']
            else:
                self.stim_vec[c] = np.ones(params['sim_time']) * params['dc']
            self.mc[c].init_vector_stim(
                np.linspace(0,
                            params['sim_time'],
                            params['sim_time']),
                self.stim_vec[c])
            pc.source_var(
                self.mc[c].tuft1(0.5)._ref_v,
                offset + c,
                sec=self.mc[c].tuft1)

        self.glu_cons = [[0 for i in xrange(cell_num)]
                         for i in xrange(cell_num)]
        self.gaba_cons = [[0 for i in xrange(cell_num)]
                          for i in xrange(cell_num)]

        self.glu_syns = [[0 for i in xrange(cell_num)]
                         for i in xrange(cell_num)]
        self.gaba_syns = [[0 for i in xrange(cell_num)]
                          for i in xrange(cell_num)]

        self.gap = [[0 for i in xrange(cell_num)] for i in xrange(cell_num)]
        self.pc = pc
        for c1 in range(cell_num):
            for c2 in range(cell_num):
                dist = abs(c1 - c2)
                if params['glu_on']:
                    self.glu_syns[c1][c2] = neuron.h.ExpSyn(
                        self.mc[c2].tuft1(0.5))
                    self.glu_syns[c1][c2].tau = params['glu_syn_tau']
                    self.glu_cons[c1][c2] = neuron.h.NetCon(
                        self.mc[c1].soma(0.5)._ref_v,
                        self.glu_syns[c1][c2],
                        25.0,
                        params['glu_delay'],
                        params['glu_weight'],
                        sec=self.mc[c1].soma)
                if params['gaba_on']:
                    self.gaba_syns[c1][c2] = neuron.h.ExpSyn(
                        self.mc[c2].tuft1(0.5))
                    self.gaba_syns[c1][c2].tau = params['gaba_syn_tau']
                    self.gaba_syns[c1][c2].e = params['gaba_syn_e']
                    self.gaba_cons[c1][c2] = neuron.h.NetCon(
                        self.mc[c1].soma(0.5)._ref_v,
                        self.gaba_syns[c1][c2],
                        25.0,
                        params['gaba_delay'],
                        params['gaba_weight'],
                        sec=self.mc[c1].soma)

                if dist > 0:

                    self.gap[c1][c2] = neuron.h.gap3(self.mc[c2].tuft1(0.5))
                    pc.target_var(
                        self.gap[c1][c2],
                        self.gap[c1][c2]._ref_vgap,
                        offset + c1)  
                    self.gap[c1][c2].g = params['gap_g_base']
        self.spk_t = [0] * cell_num
        self.apc = [0] * cell_num
        for c in range(cell_num):
            self.apc[c] = neuron.h.APCount(self.mc[c].soma(0.5))
            self.spk_t[c] = neuron.h.Vector()
            self.apc[c].thresh = 0.

            self.apc[c].record(self.spk_t[c])

    def spk_times_obj(self):
        tobj = np.zeros((self.cell_num,), dtype=np.object)

        for c in range(self.cell_num):
            tobj[c] = np.array(self.spk_t[c])
        return tobj

    def set_gaba_weight(self, gaba_weight):
        for c1 in range(self.cell_num):
            for c2 in range(self.cell_num):
                try:
                    self.gaba_cons[c1][c2].weight[0] = gaba_weight
                except:
                    print 'No GABA synapses'

        if hasattr(self, 'extra_gaba_cons_in'):
            for c in range(len(self.extra_gaba_cons_in)):
                self.extra_gaba_cons_in[c].weight[0] = gaba_weight
            for c in range(len(self.extra_gaba_cons_out)):
                self.extra_gaba_cons_out[c].weight[0] = gaba_weight

    def set_glu_weight(self, glu_weight):
        for c1 in range(self.cell_num):
            for c2 in range(self.cell_num):
                try:
                    self.glu_cons[c1][c2].weight[0] = glu_weight
                except:
                    print 'No GLU synapses'

        if hasattr(self, 'extra_glu_cons_in'):
            for c in range(len(self.extra_glu_cons_in)):
                self.extra_glu_cons_in[c].weight[0] = glu_weight
            for c in range(len(self.extra_glu_cons_out)):
                self.extra_glu_cons_out[c].weight[0] = glu_weight

    def set_gap_weight(self, gap_weight):
        for c1 in range(self.cell_num):
            for c2 in range(self.cell_num):
                try:
                    self.gap[c1][c2].g = gap_weight
                except:
                    print 'No gaps'

        if hasattr(self, 'extra_gap_in'):
            for c in range(len(self.extra_gap_in)):
                self.extra_gap_in[c].g = gap_weight
            for c in range(len(self.extra_gap_out)):
                self.extra_gap_out[c].g = gap_weight

    def set_CAn(self, can1, can2):
        for c in range(self.cell_num):
            self.mc[c].tuft1.gbar_CAn = can1
            self.mc[c].tuft2.gbar_CAn = can2

    def set_nadp(self, nadp_factor):
        neuron.h.k1_nadp *= nadp_factor
        neuron.h.k2_nadp *= nadp_factor
        neuron.h.k3_nadp *= nadp_factor
        neuron.h.k4_nadp *= nadp_factor

    def set_ek(self, ek):
        for c in range(self.cell_num):
            for sec in self.mc[c].cell_secs:
                sec.ek = ek

    def connect(self, targetGlom, cell_id, offset):
        self.extra_glu_cons_out = [0 for i in xrange(targetGlom.cell_num)]
        self.extra_glu_cons_in = [0 for i in xrange(targetGlom.cell_num)]

        self.extra_gaba_cons_out = [0 for i in xrange(targetGlom.cell_num)]
        self.extra_gaba_cons_in = [0 for i in xrange(targetGlom.cell_num)]

        self.extra_glu_syns_out = [0 for i in xrange(targetGlom.cell_num)]
        self.extra_glu_syns_in = [0 for i in xrange(targetGlom.cell_num)]

        self.extra_gaba_syns_out = [0 for i in xrange(targetGlom.cell_num)]
        self.extra_gaba_syns_in = [0 for i in xrange(targetGlom.cell_num)]

        self.extra_gap_out = [0 for i in xrange(targetGlom.cell_num)]
        self.extra_gap_in = [0 for i in xrange(targetGlom.cell_num)]

        self.pc.source_var(
            self.mc[cell_id].tuft2(0.5)._ref_v,
            offset,
            sec=self.mc[cell_id].tuft2)
        for c in range(targetGlom.cell_num):
            if self.params['glu_on']:
                self.extra_glu_syns_out[c] = neuron.h.ExpSyn(
                    targetGlom.mc[c].tuft1(0.5))
                self.extra_glu_syns_out[c].tau = self.params['glu_syn_tau']
                self.extra_glu_cons_out[c] = neuron.h.NetCon(
                    self.mc[cell_id].soma(0.5)._ref_v,
                    self.extra_glu_syns_out[c],
                    25.0,
                    self.params['glu_delay'],
                    self.params['glu_weight'],
                    sec=self.mc[cell_id].soma)
                self.extra_glu_syns_in[c] = neuron.h.ExpSyn(
                    self.mc[cell_id].tuft2(0.5))
                self.extra_glu_syns_in[c].tau = self.params['glu_syn_tau']
                self.extra_glu_cons_in[c] = neuron.h.NetCon(
                    targetGlom.mc[c].soma(0.5)._ref_v,
                    self.extra_glu_syns_in[c],
                    25.0,
                    self.params['glu_delay'],
                    self.params['glu_weight'],
                    sec=targetGlom.mc[c].soma)
            if self.params['gaba_on']:
                self.extra_gaba_syns_out[c] = neuron.h.ExpSyn(
                    targetGlom.mc[c].tuft1(0.5))
                self.extra_gaba_syns_out[c].tau = self.params['gaba_syn_tau']
                self.extra_gaba_syns_out[c].e = self.params['gaba_syn_e']
                self.extra_gaba_cons_out[c] = neuron.h.NetCon(
                    self.mc[cell_id].soma(0.5)._ref_v,
                    self.extra_gaba_syns_out[c],
                    25.0,
                    self.params['gaba_delay'],
                    self.params['gaba_weight'],
                    sec=self.mc[cell_id].soma)
                self.extra_gaba_syns_in[c] = neuron.h.ExpSyn(
                    self.mc[cell_id].tuft2(0.5))
                self.extra_gaba_syns_in[c].tau = self.params['glu_syn_tau']
                self.extra_gaba_syns_in[c].e = self.params['gaba_syn_e']
                self.extra_gaba_cons_in[c] = neuron.h.NetCon(
                    targetGlom.mc[c].soma(0.5)._ref_v,
                    self.extra_gaba_syns_in[c],
                    25.0,
                    self.params['gaba_delay'],
                    self.params['gaba_weight'],
                    sec=targetGlom.mc[c].soma)

            self.extra_gap_out[c] = neuron.h.gap3(targetGlom.mc[c].tuft1(0.5))
            self.pc.target_var(
                self.extra_gap_out[c],
                self.extra_gap_out[c]._ref_vgap,
                offset)

            self.extra_gap_out[c].g = self.params['gap_g_base']

            self.extra_gap_in[c] = neuron.h.gap3(self.mc[cell_id].tuft2(0.5))
            self.pc.target_var(
                self.extra_gap_in[c],
                self.extra_gap_in[c]._ref_vgap,
                offset + 1 + c)
            self.pc.source_var(
                targetGlom.mc[c].tuft1(0.5)._ref_v,
                offset + 1 + c,
                sec=targetGlom.mc[c].tuft1)
            self.extra_gap_in[c].g = self.params['gap_g_base']

    def glom_stim(self, delay, duration, amp):

        self.stim += [[0] * self.cell_num]
        for c in range(self.cell_num):
            self.stim[-1][c] = neuron.h.IClamp(self.mc[c].soma(0.5))
            self.stim[-1][c].delay = delay
            self.stim[-1][c].dur = duration
            self.stim[-1][c].amp = amp
