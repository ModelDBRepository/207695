"""
(C) Asaph Zylbertal 01.10.16, HUJI, Jerusalem, Israel

Basic model functions: stimulation and recording
If you use this model in your research please cite:
****************

"""

import numpy as np
import neuron


class mitral_neuron(object):

    def __del__(self):
        self.soma = None
        self.basl = None
        self.apic1 = None
        self.tuft1 = None
        self.apic2 = None
        self.tuft2 = None
        self.hlck = None
        self.iseg = None
        self.axon = None
        self.root = None

        neuron.h("forall delete_section()")

    # Application of a square pulse stimulation

    def init_square_stim(self, amp):

        stim = neuron.h.IClamp(self.root(0.5))

        stim.delay = 100
        stim.amp = amp
        stim.dur = 400
        self.stim = stim

    # Application of pulse train stimulation

    def init_train_stim(self, delay, duration, freq,
                        pulse_duration, amp, dc, limit_dc=False, noise_stdv=0):
        stim = []
        pulse_num = int(duration * freq)

        for i in range(pulse_num):

            stim.append(neuron.h.IClamp(self.root(0.5)))
            stim[i].delay = delay + i / freq
            stim[i].amp = amp
            stim[i].dur = pulse_duration
        stim.append(neuron.h.IClamp(self.root(0.5)))
        stim[i + 1].amp = dc
        if limit_dc:
            stim[i + 1].delay = delay
            stim[i + 1].dur = duration
        else:
            stim[i + 1].delay = 0
            stim[i + 1].dur = self.sim_time

        if noise_stdv > 0:

            stim.append(neuron.h.IClamp(self.root(0.5)))

            stim[i + 2].delay = 0
            stim[i + 2].dur = self.sim_time

            noise_t = np.linspace(0, self.sim_time, self.sim_time)
            t_vec = neuron.h.Vector(noise_t)
            noise_vec = np.random.normal(0, noise_stdv, self.sim_time)
            self.nstim_vec = neuron.h.Vector(noise_vec)
            self.nstim_vec.play(stim[i + 2]._ref_amp, t_vec)

        self.stim = stim

    # Record voltage from a specific segment

    def init_recording(self, seg):

        self.rec_v = self.init_vec_recording(seg._ref_v)
        self.rec_t = self.init_vec_recording(neuron.h._ref_t)

    # Record arbitrary time series

    def init_vec_recording(self, ref):
        vec = neuron.h.Vector()
        vec.record(ref)
        return vec

    def init_vector_stim(self, t, vec):

        self.t_vec = neuron.h.Vector(t)
        self.stim_vec = neuron.h.Vector(vec)
        self.stim = neuron.h.IClamp(self.root(0.5))
        self.stim.delay = 0
        self.stim.dur = self.t_vec.x[len(self.t_vec.x) - 1]
        self.stim_vec.play(self.stim._ref_amp, self.t_vec)

    def stop_recording(self):

        if hasattr(self, 'rec_v'):
            del self.rec_v
        if hasattr(self, 'rec_t'):
            del self.rec_t
        if hasattr(self, 'rec_f'):
            del self.rec_f

    # Run the model untill a steady state is reached

    def init_steady_state(
            self, test_seg, init_run_chunk=500., min_slope=0.001, max_run=2000000.):

        v = neuron.h.Vector()
        v.record(test_seg._ref_v)
        t = neuron.h.Vector()
        t.record(neuron.h._ref_t)

        self.steady = neuron.h.SaveState()    # define state object

        if self.cv.active() == 1:
            self.cv.re_init()

        neuron.h.finitialize(self.E)
        neuron.h.fcurrent()

        good_chunk = False
        failed = False

        chunks_so_far = 0
        chunk_start = 0
        while ((not good_chunk) and (not failed)):

            run_point = init_run_chunk * (chunks_so_far + 1)
            chunks_so_far += 1

            neuron.run(run_point)

            ta = np.array(t)[chunk_start:]
            va = np.array(v)[chunk_start:]
            try:

                has_spikes = np.max(va) > 30

            except:
                has_spikes = False
            if len(va) > 1:
                slope = abs(va[0] - va[-1:]) / init_run_chunk
                chunk_start = chunk_start + len(ta) + 1

                if (slope < min_slope) and (not has_spikes) and (len(ta) > 1):
                    good_chunk = True
                if (run_point > max_run):
                    failed = True

        del v
        del t
        del ta

        self.steady.save()
        return (run_point)

    def run_model(self):

        neuron.h.t = 0.
        if self.cv.active() == 1:
            self.cv.re_init()

        neuron.run(self.sim_time)
