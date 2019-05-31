"""
(C) Asaph Zylbertal 1.10.2016, HUJI, Jerusalem, Israel

Defenition of the mitral cell model used in the article
If you use this model in your research please cite:

****************

"""


from mitral_basic import mitral_neuron
import neuron
import numpy as np
import pickle
import copy
import sys


class full_mitral_neuron(mitral_neuron):

    def __init__(
        self, rest_state_file=None, num=None, ican_factor=1.0, rm_factor=1.0, nadp_factor=1.0, ncx_factor=1.0,
                 rel_rnd_factor=0.0, rel_morph_factor=0.0,
                 mv_rnd_range=0., external_params=None, rest_vals=None, rndseed=None):

        if rndseed is None:
            rndseed = np.random.randint(0, 4294967295)

        np.random.seed(rndseed)

        if rel_rnd_factor == 0.0:
            randomize_params = False
        else:
            randomize_params = True

        if rel_morph_factor == 0:
            randomize_morphology = False
        else:
            randomize_morphology = True

        self.cv = neuron.h.CVode()
        self.cv.active(1)
        self.cv.atol(0.005)

        self.nadp_factor = nadp_factor

        # load rest state file if given
        if not rest_state_file is None:
            self.rest_state_file = rest_state_file
            rest_file = open(rest_state_file, 'r')
            self.rest_vals = pickle.load(rest_file)
            rest_file.close()
            self.fih = neuron.h.FInitializeHandler(1, self.restore_states)

        if not rest_vals is None:
            self.rest_vals = rest_vals
            self.fih = neuron.h.FInitializeHandler(1, self.restore_states)

        nsegs = {
            'soma': 1,
            'basl': 1,
            'apic1': 1,
            'tuft1': 1,
            'apic2': 1,
            'tuft2': 1,
            'hlck': 1,
            'iseg': 1,
            'axon': 1}
        params = {'areas': {'apic1': 2203.6588767931335,    # membrane surface area of each compartment, based on reconstruction (um^2)
                            'apic2': 9417.240379489775,
                            'basl': 717.1285065218598,
                            'soma': 997.211443262188,
                            'tuft1': 6035.798743381776,
                            'tuft2': 1752.534970400226},
                  'axon_prop': {'axon_d': 2.0831975951069794,   # axonal compartments length and diameter, based on reconstruction (um)
                                'axon_l': 294.2832603047306,
                                'hlck_d': 3.2509999275207515,
                                'hlck_l': 9.468855108497758,
                                'iseg_d': 2.90965165346303,
                                'iseg_l': 23.747105877241342},
                  'ls': {'apic1': 107.81407122521124,       # length of each compartment, based on geometry lumping step (um)
                         'apic2': 361.44099844248615,
                         'basl': 150.65762304708946,
                         'soma': 21.008040259849658,
                         'tuft1': 633.93328472940618,
                         'tuft2': 500.00001683380111},
                  'ras': {'apic1': 217.31426241951729,      # axial resistance of each compartment, based on geometry lumping step (ohm*cm)
                          'apic2': 399.99640279865821,
                          'axon': 58.8366908030129,
                          'basl': 63.032089044574661,
                          'hlck': 58.8366908030129,
                          'iseg': 58.8366908030129,
                          'soma': 17.936577064084748,
                          'tuft1': 81.196148904926218,
                          'tuft2': 44.109855426282152},

                  'e_pas': -50.374625040653285,
                  # leak channels equilibrium potential - Eleak (mV)
                  'ek': -87.442393003543842,
                  # K+ equilibrium potential - Ek (mV)
                  'ena': 59.251437490569053,
                  # initial Na+ equilibrium potential - ENa (mV)
                  'rm': 69120.845688267262 * rm_factor,
                  # specific membrane resistance with respect to leak - rm
                  # (ohm*cm^2)

                  'soma_gbar_kfast': 4.9476362033418388e-06,
                  # somatic fast K+ channel density - gkf(soma) (S/cm^2)
                  'soma_gbar_kslow': 5.2556053969950284e-05,
                  # somatic slow K+ channel density - gks(soma) (S/cm^2)
                  'soma_gbar_nat': 0.037487574392243192,
                  # somatic transient Na+ channel density - gnat(soma)
                  # (S/cm^2)
                  'soma_vshift_nat': 9.9689951286771112,
                  # somatic transient Na+ activation curve shift  -
                  # Vshiftnat(soma) (mV)
                  'TotalPump_nadp_soma': 4.0486529543650893e-11 * nadp_factor,
                  # somatic Na+ - K+ pump density - [NaKPump](soma)
                  # (mol/cm^2)

                  'hillock_gbar_kfast': 0.0029577688816268405,
                  # axon hillock fast K+ channel density - gkf(hillock)
                  # (S/cm^2)
                  'hillock_gbar_kslow': 0.29451586142251185,
                  # axon hillock slow K+ channel density - gks(hillock)
                  # (S/cm^2)
                  'hillock_gbar_nat': 0.45435222047885954,
                  # axon hillock transient Na+ channel density -
                  # gnat(hillock) (S/cm^2)
                  'hillock_vshift_nat': 16.072461085946312,
                  # axon hillock transient Na+ activation curve shift -
                  # Vshiftnat(hillock) (mV)
                  'TotalPump_nadp_hlck': 4.3117297385288452e-12 * nadp_factor,
                  # axon hillock Na+ - K+ pump density - [NaKPump](hillock)
                  # (mol/cm^2)

                  'iseg_gbar_kfast': 0.022104023881888212,
                  # initial segment fast K+ channel density - gkf(AIS)
                  # (S/cm^2)
                  'iseg_gbar_kslow': 0.16841452722880534,
                  # initial segment slow K+ channel density - gks(AIS)
                  # (S/cm^2)
                  'iseg_gbar_nat': 1.9412301196631037,
                  # initial segment transient Na+ channel density -
                  # gnat(AIS) (S/cm^2)
                  'iseg_vshift_nat': 19.170234429829282,
                  # initial segment transient Na+ activation curve shift -
                  # Vshiftnat(AIS) (mV)
                  'TotalPump_nadp_iseg': 6.7174784489599023e-17 * nadp_factor,
                  # initial segment Na+ - K+ pump density - [NaKPump](AIS)
                  # (mol/cm^2)

                  'TotalPump_nadp_axon': 5.000911023710617e-13 * nadp_factor,
                  # passive axon Na+ - K+ pump density - [NaKPump](axon)
                  # (mol/cm^2)

                  'dend_vshift_nat': 10.908767825477128,
                  # global dendritic transient Na+ channel activation curve
                  # shift - Vshiftnat(dendrites) (mV)
                  'dendfactor': 1.0999812070427781,
                  # global dendritic surface area multiplier - DF
                  'TotalPump_nadp_dend1': 3.47266979981989e-16 * nadp_factor,
                  # global dendrite #1 Na+ - K+ pump density -
                  # [NaKPump](apical1,tuft1) (mol/cm^2)
                  'TotalPump_nadp_dend2': 1.238922334933392e-18 * nadp_factor,
                  # global dendrite #2 Na+ - K+ pump density -
                  # [NaKPump](apical2,tuft2) (mol/cm^2)

                  'apic_gbar_nat1': 0.001457764820115005,
                  # apical dendrite #1 transient Na+ channel density -
                  # gnat(apical 1) (S/cm^2)
                  'apic_gbar_nat2': 0.01586007528168524,
                  # apical dendrite #2 transient Na+ channels density -
                  # gnat(apical 1) (S/cm^2)

                  'tuft_gbar_kfast': 0.0036835938796915,
                  # both tufts fast K+ channel density - gkf(tufts)
                  # (S/cm^2)
                  'tuft_gbar_kslow': 0.0039675060799127405,
                  # both tufts slow K+ channel density - gks(tufts)
                  # (S/cm^2)
                  'tuft_gbar_nat1': 0.013937252786825303,
                  # tuft #1 transient Na+ channel density - gnat(tuft 1)
                  # (S/cm^2)
                  'tuft_gbar_nat2': 0.0005677240027995327,
                  # tuft #2 transient Na+ channel density - gnat(tuft 2)
                  # (S/cm^2)
                  'tuft1_gbar1_ican': 6.4175826111700689e-06 * ican_factor,
                  # tuft #1 Ican channel density - gcan(tuft 1) (S/cm^2)
                  'tuft2_gbar1_ican': 7.2647710811827175e-10 * ican_factor,
                  # tuft #2 Ican channel density - gcan(tuft 2) (S/cm^2)
                  'gbar_CAn1': 0.023548451822287694,
                  # tuft #1 Ca2+ channel density - gcat(tuft 1) (S/cm^2)
                  'gbar_CAn2': 0.0069060607590831187,
                  # tuft #2 Ca2+ channel density - gcat(tuft 2) (S/cm^2)
                  'TotalPump1': 1.782682527828808e-11,
                  # tuft #1 Ca2+ pump density - [PMCA](tuft 1) (mol/cm^2)
                  'TotalPump2': 5.2351815702517805e-12,
                  # tuft #2 Ca2+ pump density - [PMCA](tuft 2) (mol/cm^2)
                  'imax_ncx1': 16.485470920082754 * ncx_factor,
                  # tuft #1 Na+ - Ca2+ exchanger maximum current -
                  # INCX(max)(tuft 1) (mA/cm^2)
                  'imax_ncx2': 17.933179937488237 * ncx_factor,
                  # tuft #2 Na+ - Ca2+ exchanger maximum current -
                  # INCX(max)(tuft 2) (mA/cm^2)

                  'timefactor_h_nat': 1.2233999921732934,
                  # global transient Na+ channel h particle time constatn
                  # multiplier - tau_h,Na*
                  'timefactor_m_nat': 0.73656601116512754,
                  # global transient Na+ channel m particle time constatn
                  # multiplier - tau_m,Na*

                  'timefactor_n_kfast': 0.11817996426017603,
                  # global fast K+ channel n particle time constant
                  # multiplier - tau_n,K(fast)*
                  'vshift_kfast': 6.911190701312437,
                  # global fast K+ channel activation curve shift -
                  # Vshiftkf (mV)

                  'vshift_kslow': 42.619586356966849,
                  # global slow K+ channel activation curve shift -
                  # Vshiftks (mV)

                  'erev_ican': 15.0,
                  # Ican reversal potential - Ecan (mv)
                  'cac1_ican': 9.305069595256222e-05,
                  # Ican half activation - CAN1/2 (mM)
                  'caix_ican': 5.0908112859229471,
                  # Ican hill coefficient - phi

                  'k1_nadp': 1.8412016272811358,
                  # Na+ - K+ pump internal binding forward rate constant -
                  # k1 (/mM*ms)
                  'k2_nadp': 0.021683450354786263,
                  # Na+ - K+ pump internal binding backward rate constant -
                  # k2 (/ms)
                  'k3_nadp': 2.313154243342979,
                  # Na+ - K+ pump external binding forward rate constant -
                  # k3 (/mM*ms)
                  'DNa': 0.091535344953018627,
                  # Na+ diffusion constant - diffNa+ (um/ms)

                  'ki_CAn': 11.656113162118858,
                  # Ca2+ channel Ca2+-dependent half inactivation - ki (mM)
                  'timefactor_h_CAn': 17.57247057366412,
                  # Ca2+ channel h particle time constant multiplier -
                  # tau_h,Ca*
                  'vshift_CAn': 10.268620433693597,
                  # Ca2+ channel activation curve shift - Vshiftca (mV)

                  'ca0': 3.9774752888574831e-05,
                  # Ca2+ equilibrium concentration for the PMCA - [Ca2+]i*
                  # (mM)
                  'ca_diffusion': 0.099446021986978619,
                  # Ca2+ diffusion constant - diffCa2+ (um/ms)

                  'EndBufferKd': 0.59738060516591585,
                  # Endogenous Ca2+ buffer dissociation constant - kdend
                  # (mM)
                  'TotalEndBuffer': 10.125139872858576,
                  # Total endogenous Ca2+ buffer - [EndBufferTot] (mM)
                  'fl_ratio_ogb1': 8.3007469874154634,
                  # OGB-1 bound/unbound fluorescence ratio - uf
                  'ogb1kd': 0.00023794564736718269,
                  # OGB-1 (exogenous Ca2+ buffer) dissociation constant -
                  # kdex (mM)

                  'kca_ncx': 0.33338767394994995,
                  # Na+ - Ca2+ exchanger Ca2+ affinity - km(Ca) (mM)
                  'kna_ncx': 219.84912593958077,
                  # Na+ - Ca2+ exchanger Na+ affinity - km(Na) (mM)
                  'ksat_ncx': 0.078400175068710276,
                  # Na+ - Ca2+ exchanger saturation - ksat
                  'gamma_ncx': 0.29238768554931804,
                  # Na+ - Ca2+ exchanger voltage dependence - gamma

                  'filt_order': 6.,
                  # fluorescence results filter order
                  'time_shift': 100.}                           # fluorescence results time shift (msec)

        if randomize_params:
            for param in ['TotalPump_nadp_soma',
                          'TotalPump_nadp_hlck',
                          'TotalPump_nadp_iseg',
                          'TotalPump_nadp_axon', 'dendfactor', 'TotalPump_nadp_dend1', 'TotalPump_nadp_dend2',
                          'tuft_gbar_nat1', 'tuft_gbar_nat2',
                          'imax_ncx1', 'imax_ncx2',
                          'k1_nadp', 'k2_nadp', 'k3_nadp',
                          'kca_ncx', 'kna_ncx', 'ksat_ncx', 'gamma_ncx']:
                params[param] = randomize_rel(params[param], rel_rnd_factor)
            for param in [
                'soma_vshift_nat', 'hillock_vshift_nat', 'iseg_vshift_nat', 'dend_vshift_nat',
                          'vshift_kfast', 'vshift_kslow', 'erev_ican', 'vshift_CAn']:
                params[param] = randomize_voltage(params[param], mv_rnd_range)

        if randomize_morphology:
            for prop in ['ls', 'areas']:
                for comp in ['soma', 'basl', 'apic1', 'apic2', 'tuft1', 'tuft2']:
                    params[prop][comp] = randomize_rel(
                        params[prop][comp],
                        rel_morph_factor)

        self.params = params
        if not (external_params is None):
            self.params = external_params
            params = external_params

        # enable section name suffix to allow multiple cells
        if not num is None:
            suf = '_%d' % (num)
            self.num = num
        else:
            suf = ''

        neuron.h.celsius = 23.0

        # create sections and set geometry

        self.soma = neuron.h.Section(name='soma' + suf)
        self.soma.Ra = params['ras']['soma']
        self.soma.L = params['ls']['soma']
        self.soma.diam = params['areas'][
            'soma'] / (np.pi * params['ls']['soma'])
        self.soma.nseg = nsegs['soma']

        self.hlck = neuron.h.Section(name='hlck' + suf)
        self.hlck.Ra = params['ras']['hlck']
        self.hlck.L = params['axon_prop']['hlck_l']
        self.hlck.nseg = nsegs['hlck']
        self.hlck.diam = params['axon_prop']['hlck_d']

        self.iseg = neuron.h.Section(name='iseg' + suf)
        self.iseg.Ra = params['ras']['iseg']
        self.iseg.L = params['axon_prop']['iseg_l']
        self.iseg.nseg = nsegs['iseg']
        self.iseg.diam = params['axon_prop']['iseg_d']

        self.axon = neuron.h.Section(name='axon' + suf)
        self.axon.Ra = params['ras']['axon']
        self.axon.L = params['axon_prop']['axon_l']
        self.axon.nseg = nsegs['axon']
        self.axon.diam = params['axon_prop']['axon_d']

        self.basl = neuron.h.Section(name='basl' + suf)
        self.basl.Ra = params['ras']['basl']
        self.basl.L = params['ls']['basl']
        self.basl.diam = params['areas'][
            'basl'] / (np.pi * params['ls']['basl'])
        self.basl.nseg = nsegs['basl']

        self.apic1 = neuron.h.Section(name='apic1' + suf)
        self.apic1.Ra = params['ras']['apic1']
        self.apic1.L = params['ls']['apic1']
        self.apic1.diam = params['areas'][
            'apic1'] / (np.pi * params['ls']['apic1'])
        self.apic1.nseg = nsegs['apic1']

        self.tuft1 = neuron.h.Section(name='tuft1' + suf)
        self.tuft1.Ra = params['ras']['tuft1']
        self.tuft1.L = params['ls']['tuft1']
        self.tuft1.diam = params['areas'][
            'tuft1'] / (np.pi * params['ls']['tuft1'])
        self.tuft1.nseg = nsegs['tuft1']

        self.apic2 = neuron.h.Section(name='apic2' + suf)
        self.apic2.Ra = params['ras']['apic2']
        self.apic2.L = params['ls']['apic2']
        self.apic2.diam = params['areas'][
            'apic2'] / (np.pi * params['ls']['apic2'])
        self.apic2.nseg = nsegs['apic2']

        self.tuft2 = neuron.h.Section(name='tuft2' + suf)
        self.tuft2.Ra = params['ras']['tuft2']
        self.tuft2.L = params['ls']['tuft2']
        self.tuft2.diam = params['areas'][
            'tuft2'] / (np.pi * params['ls']['tuft2'])
        self.tuft2.nseg = nsegs['tuft2']

        # connect sections

        self.basl.connect(self.soma, 0.5, 0)
        self.apic1.connect(self.soma, 1, 0)
        self.tuft1.connect(self.apic1, 1, 0)
        self.apic2.connect(self.soma, 1, 0)
        self.tuft2.connect(self.apic2, 1, 0)
        self.hlck.connect(self.soma, 0.5, 0)
        self.iseg.connect(self.hlck, 1, 0)
        self.axon.connect(self.iseg, 1, 0)

        self.cell_secs = [
            self.axon,
            self.hlck,
            self.iseg,
            self.soma,
            self.apic1,
            self.tuft1,
            self.apic2,
            self.tuft2,
            self.basl]

        self.E = params['e_pas']
        self.sim_time = 0.6e3

        self.root = self.soma

        self.tot_seg = 0

        for sec in self.cell_secs:
            self.tot_seg += sec.nseg

        for sec in self.cell_secs:

            # insert leak channels and Na+ - K+ pump to all sections

            sec.insert('kleak')
            sec.insert('naleak')
            sec.insert('nadp')
            sec.cm = 1 * \
                params['dendfactor']  # capacitance is 1 uF/cm^2 * dendfactor

            if not (sec == self.axon or sec == self.basl):
                # insert active channels (fast K+, slow K+, transient Na+) to
                # all section except axon and basl dendrite

                sec.insert('nat')
                sec.insert('kslow')
                sec.insert('kfast')

                # set global activation curve shifts and time constant
                # multipliers

                sec.vshift_kfast = params['vshift_kfast']
                sec.vshift_kslow = params['vshift_kslow']
                sec.timefactor_n_kfast = params['timefactor_n_kfast']
                sec.timefactor_m_nat = params['timefactor_m_nat']
                sec.timefactor_h_nat = params['timefactor_h_nat']

                # set K+ equilibrium potential

                sec.ek = params['ek']

            for seg in sec:

                # calculate leak conductance, use dendfactor for all processes
                if sec == self.soma:
                    sec_g = 1. / params['rm']
                else:
                    sec_g = params['dendfactor'] / params['rm']

                # use K+ equilibrium potential, initial Na+ equilibrium
                # potential and leak reversal potential to divide leak
                # conductance between K+ and Na+
                seg.g_kleak = sec_g / \
                    (1 + ((params['ek'] - params['e_pas']) / (
                        params['e_pas'] - params['ena'])))
                seg.g_naleak = sec_g - seg.g_kleak

        self.soma.cm = 1.0        # Soma capacitance is 1 uF/cm^2

        for sec in [self.tuft1, self.tuft2]:

            # insert calcium mechanisms to dendritic tufts
            sec.insert('CAn')
            sec.insert('cadp')
            sec.insert('ican_ns')
            sec.insert('ncx')

            # set channel parameters
            sec.gbar_kfast = params['tuft_gbar_kfast']
            sec.gbar_kslow = params['tuft_gbar_kslow']
            sec.vshiftm_CAn = params['vshift_CAn']
            sec.vshifth_CAn = params['vshift_CAn']
            sec.timefactor_h_CAn = params['timefactor_h_CAn']

        self.apic1.gbar_nat = params['apic_gbar_nat1']
        self.apic2.gbar_nat = params['apic_gbar_nat2']
        self.tuft1.gbar_nat = params['tuft_gbar_nat1']
        self.tuft2.gbar_nat = params['tuft_gbar_nat2']

        for sec in [self.apic1, self.tuft1, self.apic2, self.tuft2]:
            sec.vshift_nat = params['dend_vshift_nat']

        self.root.push()
        neuron.h.distance()
        base_dist = neuron.h.distance(1.0)
        neuron.h.pop_section()

        # set apical dendrite K+ channel densities as a gradient between
        # somatic and tuft densities (or only one average value when nseg=1)

        self.apic1.push()
        for seg in self.apic1:
            prop = (neuron.h.distance(seg.x) - base_dist) / self.apic1.L
            seg.gbar_kfast = params[
                'tuft_gbar_kfast'] * prop + params[
                    'soma_gbar_kfast'] * (
                        1 - prop)
            seg.gbar_kslow = params[
                'tuft_gbar_kslow'] * prop + params[
                    'soma_gbar_kslow'] * (
                        1 - prop)

        neuron.h.pop_section()

        self.apic2.push()
        for seg in self.apic2:
            prop = (neuron.h.distance(seg.x) - base_dist) / self.apic2.L
            seg.gbar_kfast = params[
                'tuft_gbar_kfast'] * prop + params[
                    'soma_gbar_kfast'] * (
                        1 - prop)
            seg.gbar_kslow = params[
                'tuft_gbar_kslow'] * prop + params[
                    'soma_gbar_kslow'] * (
                        1 - prop)

        neuron.h.pop_section()

        # more channel parameters

        self.soma.gbar_nat = params['soma_gbar_nat']
        self.soma.vshift_nat = params['soma_vshift_nat']
        self.soma.gbar_kfast = params['soma_gbar_kfast']
        self.soma.gbar_kslow = params['soma_gbar_kslow']

        self.hlck.gbar_nat = params['hillock_gbar_nat']
        self.hlck.vshift_nat = params['hillock_vshift_nat']
        self.hlck.gbar_kfast = params['hillock_gbar_kfast']
        self.hlck.gbar_kslow = params['hillock_gbar_kslow']

        self.iseg.gbar_nat = params['iseg_gbar_nat']
        self.iseg.vshift_nat = params['iseg_vshift_nat']
        self.iseg.gbar_kfast = params['iseg_gbar_kfast']
        self.iseg.gbar_kslow = params['iseg_gbar_kslow']

        self.tuft1.imax_ncx = params['imax_ncx1']
        self.tuft1.gbar_CAn = params['gbar_CAn1']
        self.tuft1.TotalPump_cadp = params['TotalPump1']

        self.tuft2.imax_ncx = params['imax_ncx2']
        self.tuft2.gbar_CAn = params['gbar_CAn2']
        self.tuft2.TotalPump_cadp = params['TotalPump2']

        self.tuft1.gbar1_ican_ns = params['tuft1_gbar1_ican']
        self.tuft2.gbar1_ican_ns = params['tuft2_gbar1_ican']

        self.soma.TotalPump_nadp = params['TotalPump_nadp_soma']
        self.hlck.TotalPump_nadp = params['TotalPump_nadp_hlck']
        self.iseg.TotalPump_nadp = params['TotalPump_nadp_iseg']
        self.axon.TotalPump_nadp = params['TotalPump_nadp_axon']

        neuron.h.erev_ican_ns = params['erev_ican']
        neuron.h.cac1_ican_ns = params['cac1_ican']
        neuron.h.caix_ican_ns = params['caix_ican']

        for sec in [self.apic1, self.tuft1]:
            sec.TotalPump_nadp = params['TotalPump_nadp_dend1']

        for sec in [self.apic1, self.tuft1]:
            sec.TotalPump_nadp = params['TotalPump_nadp_dend2']

        neuron.h.k1_nadp = params['k1_nadp']
        neuron.h.k2_nadp = params['k2_nadp']
        neuron.h.k3_nadp = params['k3_nadp']

        neuron.h.DNa_nadp = params['DNa']
        neuron.h.ki_CAn = params['ki_CAn']

        neuron.h.TotalEndBuffer_cadp = params['TotalEndBuffer']
        neuron.h.k2bufend_cadp = neuron.h.k1bufend_cadp * params['EndBufferKd']
        neuron.h.DCa_cadp = params['ca_diffusion']
        neuron.h.fl_ratio_cadp = params['fl_ratio_ogb1']
        neuron.h.cai0_ca_ion = params['ca0']

        neuron.h.kna_ncx = params['kna_ncx']
        neuron.h.kca_ncx = params['kca_ncx']
        neuron.h.gamma_ncx = params['gamma_ncx']
        neuron.h.ksat_ncx = params['ksat_ncx']

        # fixed values:

        neuron.h.nao0_na_ion = 151.3          # [Na+]o
        neuron.h.cao0_ca_ion = 2.0            # [Ca2+]o
        neuron.h.TotalExBuffer_cadp = 0.05    # Total exogenous Ca2+ buffer (OGB-1)
        neuron.h.k1bufex_cadp = 200.          # OGB-1-Ca2+ binding forward rate constant
        neuron.h.k1bufend_cadp = 100.         # endogenous Ca2+ buffer binding forward rate
        neuron.h.dep_factor_cadp = 0
        neuron.h.k2bufex_cadp = params[
            'ogb1kd'] * 200.     # OGB-1-Ca2+ binding backward rate constant

        # set initial [Na]i according to initial Na+ equilibrium potential

        den = (8.314e3 * (273.15 + neuron.h.celsius)) / 9.6485e4
        neuron.h.nai0_na_ion = neuron.h.nao0_na_ion * \
            np.exp(-params['ena'] / den)

    # save all model state variables to a file

    def save_states(self, filename):
        vals = []
        for sec in self.cell_secs:
            for seg in sec:
                vals = vals + [
                    seg.v,
                    seg.nadp.pump,
                    seg.nadp.pumpna,
                    np.array(
                        seg.nadp.na)]
            if not (sec == self.axon or sec == self.basl):
                for seg in sec:
                    vals = vals + [
                        seg.nat.m,
                        seg.nat.h,
                        seg.kfast.n,
                        seg.kslow.a,
                        seg.kslow.b1,
                        seg.kslow.b]

        for sec in [self.tuft1, self.tuft2]:
            for seg in sec:
                vals = vals + [seg.CAn.m,
                               seg.CAn.h,
                               np.array(seg.cadp.ca),
                               np.array(seg.cadp.CaEndBuffer),
                               np.array(seg.cadp.CaExBuffer),
                               np.array(seg.cadp.EndBuffer),
                               np.array(seg.cadp.ExBuffer),
                               seg.cadp.pump,
                               seg.cadp.pumpca,
                               seg.ican_ns.m1,
                               seg.ican_ns.m2]

        vals = vals + [neuron.h.k4_nadp, neuron.h.k2_cadp, neuron.h.k4_cadp]
        if not filename is None:
            f = open(filename, 'w')
            pickle.dump(vals, f)
            f.close()
        return vals

    # restore model state variables from a file

    def restore_states(self):

        # f=open(self.rest_state_file, 'r')
        # vals=pickle.load(f)
        # f.close()

        vals = copy.deepcopy(self.rest_vals)
        for sec in self.cell_secs:
            for seg in sec:
                seg.v = vals.pop(0)
                seg.nadp.pump = vals.pop(0) * self.nadp_factor
                seg.nadp.pumpna = vals.pop(0) * self.nadp_factor
                nas = vals.pop(0)
                for ann in range(len(seg.nadp.na)):
                    seg.nadp.na[ann] = nas[ann]
                    seg.nai = nas[0]

            if not (sec == self.axon or sec == self.basl):
                for seg in sec:
                    seg.nat.m = vals.pop(0)
                    seg.nat.h = vals.pop(0)
                    seg.kfast.n = vals.pop(0)
                    seg.kslow.a = vals.pop(0)
                    seg.kslow.b1 = vals.pop(0)
                    seg.kslow.b = vals.pop(0)

        for sec in [self.tuft1, self.tuft2]:
            for seg in sec:
                seg.CAn.m = vals.pop(0)
                seg.CAn.h = vals.pop(0)
                cas = vals.pop(0)
                caendbuffers = vals.pop(0)
                caexbuffers = vals.pop(0)
                endbuffers = vals.pop(0)
                exbuffers = vals.pop(0)
                for ann in range(len(seg.cadp.ca)):
                    seg.cadp.ca[ann] = cas[ann]
                    seg.cai = cas[0]
                    seg.cadp.CaEndBuffer[ann] = caendbuffers[ann]
                    seg.cadp.CaExBuffer[ann] = caexbuffers[ann]
                    seg.cadp.EndBuffer[ann] = endbuffers[ann]
                    seg.cadp.ExBuffer[ann] = exbuffers[ann]

                seg.cadp.pump = vals.pop(0)
                seg.cadp.pumpca = vals.pop(0)
                seg.ican_ns.m1 = vals.pop(0)
                seg.ican_ns.m2 = vals.pop(0)

        neuron.h.k4_nadp = vals.pop(0)
        neuron.h.k2_cadp = vals.pop(0)
        neuron.h.k4_cadp = vals.pop(0)


def randomize_rel(mean, rel_dist):

    high = mean + mean * rel_dist
    low = mean - mean * rel_dist
    rng = high - low
    shift = rng * np.random.rand()
    return low + shift


def randomize_voltage(mean, abs_rng):

    rng = abs_rng * 2
    shift = rng * np.random.rand()
    return mean - abs_rng + shift
