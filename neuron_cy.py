"""
..Project: FeedbackNet
  Platform: Linux
  Description: Neuron models in Cython

..Author: Ayon Borthakur(ab2535@cornell.edu)
"""
import numpy as np
import torch
iscuda = torch.cuda.is_available()
iscuda = False
if iscuda:
    import torch.cuda as t
else:
    import torch as t


class Neuron(object):
    """Neuron class. """
    def __init__(self):
        pass


class LIFneuron(Neuron):
    """LIF neuron model with Rungekutta integration. """

    def __init__(self):
        pass


class MitralApiLIF(LIFneuron):
    """Mitral Cell LIF neuron. """

    """Mitral Cell LIF neuron. """

    def __init__(self, **kwargs):

        self.p = kwargs
        self.dt = self.p["dt"]
        self.v_th = self.p["v_th_apimi"]
        self.r_m = self.p["r_apimi"]
        self.v_spike = self.p["v_spike"]
        self.tau = self.p["tau_apimi"]
        self.t_ref = self.p["t_ref_apimi"]
        self.T = self.p["T"]
        self.amp_osc = self.p["amp_osc"]
        self.f = self.p["f"]
        self.offset_osc = self.p["offset_osc"]
        self.binL = self.p["binL"]
        self.num_apimi = self.p["num_apimi"]
        self.stat = kwargs["stat"]
        self.tstep = 0
        self.train_idx = 0
        self.r = self.p["r_apimi"]

        # if iscuda:
        self.v = t.FloatTensor(self.p["batchsize"], self.num_apimi).fill_(0)
        self.v_prev = t.FloatTensor(self.p["batchsize"], self.num_apimi).fill_(0)
        self.count = t.FloatTensor(self.p["batchsize"], self.num_apimi).fill_(0)
        self.t_on = t.FloatTensor(self.p["batchsize"], self.num_apimi).fill_(0)
        self.spike_val = t.FloatTensor(self.p["batchsize"], self.num_apimi).fill_(0)
        self.spk_time = t.FloatTensor(self.p["batchsize"], self.num_apimi).fill_(0)

    def neu_simul(self, apimi_v_th, train_data, tstep):

        if self.tstep != 0:
            self.v_prev[:] = self.v[:]
        self.tstep = tstep
        self.r = self.gamma()
        self.rungekutta_integrate(apimi_v_th, train_data)

    def gamma(self):
        """
        Computes the total shunt inhibition due to gamma wave
        to a mitral cell.
        """

        res_shunt = (self.amp_osc * np.sin((2 * np.pi * self.f * 0.001 *
                                            self.dt * self.tstep) +
                                           (np.pi / 2)) + self.offset_osc)
        res = self.r_m / res_shunt
        # self.gamma_prf[:, self.tstep] = res
        return res

    def rungekutta_integrate(self, apimi_v_th, data):

        train_data = data

        if not ((self.tstep * self.dt) % self.binL):
            self.v_prev[:, :] = 0.
            self.spike_val[:, :] = 0.
        nospk_idx = torch.nonzero(self.t_on[:, :] >= self.tstep * self.dt)
        if nospk_idx.numel():
            nospk_idx = torch.squeeze(nospk_idx, 1)
        '''
        Runge-Kutta Integration method.
        v' = (-self.v_prev + self.i_inp * (self.r))/self.tau
        '''
        k_1 = ((-self.v_prev[:, :] + train_data[:, :] * self.r) / self.tau)  #different voltages based off integration
        k_2 = ((-(self.v_prev[:, :] + ((self.dt / 2) * k_1[:, :])) +
                train_data[:, :] * self.r) / self.tau)
        k_3 = ((-(self.v_prev[:, :] + ((self.dt / 2) * k_2[:, :])) +
                train_data[:, :] * self.r) / self.tau)
        k_4 = ((-(self.v_prev[:, :] + (self.dt * k_3[:, :])) +
                train_data[:, :] * self.r) / self.tau)
        self.v[:] = (self.v_prev[:, :] + (self.dt / 6) *     # voltage 
                     (k_1[:, :] + 2 * k_2[:, :] + 2 *
                      k_3[:, :] + k_4[:, :]))

        # import pdb; pdb.set_trace()
        '''
        if nospk_idx.numel():
            self.v = self.v.index_fill_(0, nospk_idx, 0.)
        '''
        if nospk_idx.numel():
            self.v[nospk_idx[:, 0], nospk_idx[:, 1]] = 0.
        self.v_prev[:, :] = self.v[:, :]

        spk_idx = torch.nonzero(self.v[:, :] >= apimi_v_th[:])
        # spk_idx = (self.v[:] >= self.v_th).nonzero()
        if spk_idx.numel():
            spk_idx = torch.squeeze(spk_idx, 1)
            self.v_prev[spk_idx[:, 0], spk_idx[:, 1]] = 0.
            self.v[spk_idx[:, 0], spk_idx[:, 1]] = self.v[spk_idx[:, 0], spk_idx[:, 1]] + self.v_spike
            self.t_on[spk_idx[:, 0], spk_idx[:, 1]] = ((self.tstep * self.dt) + (
                    self.binL - ((self.tstep * self.dt) % self.binL)))
            self.count[spk_idx[:, 0], spk_idx[:, 1]] = self.count[spk_idx[:, 0], spk_idx[:, 1]] + 1
            self.spike_val[spk_idx[:, 0], spk_idx[:, 1]] = data[spk_idx[:, 0], spk_idx[:, 1]]
            self.spk_time[spk_idx[:, 0], spk_idx[:, 1]] = self.tstep * self.dt
            # import pdb;pdb.set_trace()
        # self.v_mi[:, self.tstep] = self.v[:]


class ETLIF(LIFneuron):
    """Mitral Cell LIF neuron. """

    def __init__(self, **kwargs):

        self.p = kwargs
        self.dt = self.p["dt"]
        self.v_th = self.p["v_th_et"]
        self.tau = self.p["tau_et"]
        self.T = self.p["T"]
        self.binL = self.p["binL"]
        self.num_et = self.p["num_et"]
        self.res = t.FloatTensor([1.])  # Need to change
        self.v_spike = self.p["v_spike"]
        self.r = t.FloatTensor(self.p["r_et"])
        self.c = t.FloatTensor([self.p["c_et"]])
        self.sensor2et = t.as_tensor(self.p["sensor2et"])
        self.et2api = t.FloatTensor(self.p["et2api"])
        self.tstep = 0
        self.train_idx = 0

        # if iscuda:
        '''
        self.v = t.FloatTensor(self.p["batchsize"], int(self.num_et/self.p["mi_dup"])).fill_(0)
        self.v_prev = t.FloatTensor(self.p["batchsize"], int(self.num_et/self.p["mi_dup"])).fill_(0)
        self.count = t.FloatTensor(self.p["batchsize"], int(self.num_et/self.p["mi_dup"]),
                                 int(self.T/self.binL)).fill_(0)
        self.spike = t.FloatTensor(self.p["batchsize"], int(self.num_et/self.p["mi_dup"])).fill_(0)
        # self.v_prf = t.FloatTensor(self.num_et, int(self.T/self.dt)).fill_(0)
        '''
        self.v = t.FloatTensor(self.p["batchsize"], int(self.num_et)).fill_(0)
        self.v_prev = t.FloatTensor(self.p["batchsize"], int(self.num_et)).fill_(0)
        self.count = t.FloatTensor(self.p["batchsize"], int(self.num_et),
                                 int(self.T / self.binL)).fill_(0)
        self.spike = t.FloatTensor(self.p["batchsize"], int(self.num_et)).fill_(0)

    def neu_simul(self, et_v_th, train_data, g_syn, tstep):

        self.tstep = tstep
        '''
        if tstep >= int(self.binL/self.dt):
            self.r = (self.res/g_syn)
            # import pdb;pdb.set_trace()
        '''
        # self.r = self.res / g_syn
        self.rungekutta_integrate(et_v_th, train_data, g_syn)

    def rungekutta_integrate(self, et_v_th, data, g_syn):

        x, y = np.divmod(self.tstep * self.dt, self.binL)
        x = int(abs(x))
        train_data = data
        train_data = train_data * self.r
        sensor_ntce2 = (train_data[:] * (self.res / (self.c * g_syn)) * self.dt)
        self.v[:] = torch.mm(sensor_ntce2, self.sensor2et)
        # self.v[:] = self.v[:] / t.transpose(t.sum(self.v[:], 1).expand(self.v[:].size()[1], -1), 0, 1)
        self.v[:] = self.v[:] * 0.625 * (self.num_et/self.p["mi_dup"])
        self.v[:] = torch.mm(self.v[:], self.et2api)
        # self.v[:] = self.v[:] / t.transpose(t.sum(self.v[:], 1).expand(self.v[:].size()[1], -1), 0, 1)
        # self.v[:] = self.v[:] * 1.
        self.count[:, :, x] = self.v[:]
        self.v_prev[:] = self.v[:]


class PGLIF(LIFneuron):
    """Mitral Cell LIF neuron. """

    def __init__(self, **kwargs):

        self.p = kwargs
        self.dt = self.p["dt"]
        self.v_th = self.p["v_th_pg"]
        # self.tau = self.p["tau_pg"]
        self.T = self.p["T"]
        self.binL = self.p["binL"]
        self.num_pg = self.p["num_pg"]
        self.v_spike = self.p["v_spike"]
        # self.r = self.p["r_pg"]
        self.tstep = 0
        self.train_idx = 0

        # if iscuda:
        self.v = t.FloatTensor(self.p["batchsize"], self.num_pg).fill_(0)
        self.v_prev = t.FloatTensor(self.p["batchsize"], self.num_pg).fill_(0)
        self.count = t.FloatTensor(self.p["batchsize"], self.num_pg,
                                 int(self.T/self.binL)).fill_(0)
        self.spike = t.FloatTensor(self.p["batchsize"], self.num_pg).fill_(0)
        # self.v_prf = t.FloatTensor(self.num_pg, int(self.T / self.dt)).fill_(0)

    def neu_simul(self, pg_v_th, pg_data, tstep):

        self.tstep = tstep
        self.rungekutta_integrate(pg_v_th, pg_data)

    def rungekutta_integrate(self, pg_v_th, pg_data):

        # import pdb;pdb.set_trace()
        # pg_data[:] = t.sum(pg_data)
        x, y = np.divmod(self.tstep * self.dt, self.binL)
        x = int(abs(x))
        self.spike[:, :] = 0
        self.v_prev[:, :] = 0
        self.v[:, :] = self.v_prev[:, :] + (pg_data[:] * self.dt)
        # self.v[:] = self.v_prev[:] + pg_data[:]
        self.count[:, :, x] = self.v[:]
        self.v_prev[:, :] = self.v[:, :]
        # self.v_prf[:, self.tstep] = self.v[:]


class MitralLIF(LIFneuron):
    """Mitral Cell LIF neuron. """

    def __init__(self, **kwargs):

        self.p = kwargs
        self.dt = self.p["dt"]
        self.v_th = self.p["v_th_mi"]
        self.r_m = self.p["r_mi"]
        self.v_spike = self.p["v_spike"]
        self.tau = self.p["tau_mi"]
        self.t_ref = self.p["t_ref_mi"]
        self.T = self.p["T"]
        self.amp_osc = self.p["amp_osc"]
        self.f = self.p["f"]
        self.offset_osc = self.p["offset_osc"]
        self.binL = self.p["binL"]
        self.num_mi = self.p["num_mi"]
        self.stat = kwargs["stat"]
        self.tstep = 0
        self.train_idx = 0
        self.r = 1.

        # if iscuda:
        self.v = t.FloatTensor(self.p["batchsize"], self.num_mi).fill_(0)
        self.v_prev = t.FloatTensor(self.p["batchsize"], self.num_mi).fill_(0)
        self.count = t.FloatTensor(self.p["batchsize"], self.num_mi).fill_(0)
        self.t_on = t.FloatTensor(self.p["batchsize"], self.num_mi).fill_(0)
        self.spike = t.FloatTensor(self.p["batchsize"], self.num_mi).fill_(0)
        self.spk_time = t.FloatTensor(self.p["batchsize"], self.num_mi).fill_(0)

        # self.v_mi = t.FloatTensor(self.num_mi, self.p["shape"]).fill_(0)
        # self.gamma_prf = t.FloatTensor(self.num_mi, int(self.T / self.dt)).fill_(0)

    def neu_simul(self, mi_v_th, train_data, tstep):

        if self.tstep != 0:
            self.v_prev[:] = self.v[:]
        self.tstep = tstep
        self.r = self.gamma()
        # self.r = self.r_m
        self.rungekutta_integrate(mi_v_th, train_data)
    

    def gamma(self):
        """
        Computes the total shunt inhibition due to gamma wave
        to a mitral cell.
        """

        res_shunt = (self.amp_osc * np.sin((2 * np.pi * self.f * 0.001 *
                                            self.dt * self.tstep) +
                                           (np.pi / 2)) + self.offset_osc)
        res = self.r_m / res_shunt
        # self.gamma_prf[:, self.tstep] = res
        return res

    def rungekutta_integrate(self, mi_v_th, data):

        train_data = data

        if not ((self.tstep * self.dt) % self.binL):
            self.v_prev[:, :] = 0.
        self.spike[:, :] = 0
        nospk_idx = torch.nonzero(self.t_on[:, :] >= self.tstep * self.dt)
        if nospk_idx.numel():
            nospk_idx = torch.squeeze(nospk_idx, 1)
        '''
        Runge-Kutta Integration method.
        v' = (-self.v_prev + self.i_inp * (self.r))/self.tau
        '''
        k_1 = ((-self.v_prev[:, :] + train_data[:, :] * self.r)/self.tau)
        k_2 = ((-(self.v_prev[:, :] + ((self.dt / 2) * k_1[:, :])) +
                train_data[:, :] * self.r)/self.tau)
        k_3 = ((-(self.v_prev[:, :] + ((self.dt / 2) * k_2[:, :])) +
                train_data[:, :] * self.r)/self.tau)
        k_4 = ((-(self.v_prev[:, :] + (self.dt * k_3[:, :])) +
                train_data[:, :] * self.r)/self.tau)
        self.v[:] = (self.v_prev[:, :] + (self.dt / 6) *
                     (k_1[:, :] + 2 * k_2[:, :] + 2 *
                      k_3[:, :] + k_4[:, :]))

        # import pdb; pdb.set_trace()
        '''
        if nospk_idx.numel():
            self.v = self.v.index_fill_(0, nospk_idx, 0.)
        '''
        if nospk_idx.numel():
            self.v[nospk_idx[:, 0], nospk_idx[:, 1]] = 0.
        self.v_prev[:, :] = self.v[:, :]

        spk_idx = torch.nonzero(self.v[:, :] >= mi_v_th[:])
        # spk_idx = (self.v[:] >= self.v_th).nonzero()
    
        if spk_idx.numel():
            
            spk_idx = torch.squeeze(spk_idx, 1)
            
            self.v_prev[spk_idx[:, 0], spk_idx[:, 1]] = 0.
            
            print(self.v.shape, spk_idx.shape, list(sorted(spk_idx[:, 0].numpy())), list(sorted(spk_idx[:, 1].numpy())), self.v_spike, spk_idx[:, 0].numpy(), spk_idx[:, 1].numpy())
            self.v[list(sorted(spk_idx[:, 0].numpy())), list(sorted(spk_idx[:, 1].numpy()))] += self.v_spike
            '''
            self.v[spk_idx[:, 0], spk_idx[:, 1]] += self.v_spike
            
            self.t_on[spk_idx[:, 0], spk_idx[:, 1]] = ((self.tstep * self.dt) + (
                    self.binL - ((self.tstep * self.dt) % self.binL)))
            
            self.count[spk_idx[:, 0], spk_idx[:, 1]] += 1
            '''
            self.spike[spk_idx[:, 0], spk_idx[:, 1]] = 1
            
            self.spk_time[spk_idx[:, 0], spk_idx[:, 1]] = self.tstep * self.dt
        
            # import pdb;pdb.set_trace()
        # self.v_mi[:, self.tstep] = self.v[:]


class GranuleLIF(LIFneuron):
    """Granule cell LIF neuron."""

    def __init__(self, **kwargs):

        self.p = kwargs
        self.dt = self.p["dt"]
        # self.v_th = self.p["v_th_gr"]
        self.r = self.p["r_gr"]
        self.v_spike = self.p["v_spike"]
        self.tau = self.p["tau_gr"]
        self.t_ref = self.p["t_ref_gr"]
        self.T = self.p["T"]
        self.en_exc = self.p["en_exc"]
        self.binL = self.p["binL"]
        self.num_gr = self.p["num_gr"]
        self.tstep = 0

        # if iscuda:
        self.v = t.FloatTensor(self.p["batchsize"], self.num_gr).fill_(0)
        self.v_prev = t.FloatTensor(self.p["batchsize"], self.num_gr).fill_(0)
        self.count = t.FloatTensor(self.p["batchsize"], self.num_gr).fill_(0)
        self.spk_time = t.FloatTensor(self.p["batchsize"], self.num_gr).fill_(0)
        self.t_on = t.FloatTensor(self.p["batchsize"], self.num_gr).fill_(0)
        self.i_syn = t.FloatTensor(self.p["batchsize"], self.num_gr).fill_(0)
        self.spike = t.FloatTensor(self.p["batchsize"], self.num_gr).fill_(0)

    def neu_simul(self, gr_v_th, gw_tot, tstep):

        if self.tstep != 0:
            self.v_prev[:, :] = self.v[:, :]
        self.tstep = tstep
        self.inp_syn_current(gw_tot)
        self.rungekutta_integrate(gr_v_th)

    def inp_syn_current(self, gw_tot):

        self.i_syn[:, :] = gw_tot[:] * (self.en_exc - self.v[:, :])

    def rungekutta_integrate(self, gr_v_th):

        if not ((self.tstep * self.dt) % self.binL):
            self.v_prev[:, :] = 0.
        self.spike[:, :] = 0
        # nospk_idx = (self.t_on[:] >= self.tstep * self.dt).nonzero()
        nospk_idx = torch.nonzero(self.t_on[:, :] >= self.tstep * self.dt)
        if nospk_idx.numel():
            nospk_idx = torch.squeeze(nospk_idx, 1)
        '''
        Runge-Kutta Integration method.
        v' = (-self.v_prev + self.i_inp * (self.r))/self.tau
        '''
        
        k_1 = ((-self.v_prev[:, :] + self.i_syn[:, :] * self.r) / self.tau)
        k_2 = ((-(self.v_prev[:, :] + ((self.dt / 2) * k_1[:, :])) +
                self.i_syn[:, :] * self.r) / self.tau)
        k_3 = ((-(self.v_prev[:, :] + ((self.dt / 2) * k_2[:, :])) +
                self.i_syn[:, :] * self.r) / self.tau)
        k_4 = ((-(self.v_prev[:, :] + (self.dt * k_3[:, :])) +
                self.i_syn[:, :] * self.r) / self.tau)
        self.v[:] = (self.v_prev[:, :] + (self.dt / 6) *
                     (k_1[:, :] + 2 * k_2[:, :] + 2 *
                      k_3[:, :] + k_4[:, :]))
        # print(self.i_syn.type())
        '''
        if nospk_idx.numel():
            self.v[:] = self.v.index_fill_(0, nospk_idx, 0.)
        '''
        if nospk_idx.numel():
            self.v[nospk_idx[:, 0], nospk_idx[:, 1]] = 0.
        self.v_prev[:, :] = self.v[:, :]
        spk_idx = torch.nonzero(self.v[:, :] >= gr_v_th[:])
        if spk_idx.numel():
            spk_idx = torch.squeeze(spk_idx, 1)
            self.v_prev[spk_idx[:, 0], spk_idx[:, 1]] = 0.
            self.v[spk_idx[:, 0], spk_idx[:, 1]] = self.v[spk_idx[:, 0], spk_idx[:, 1]] + self.v_spike
            self.t_on[spk_idx[:, 0], spk_idx[:, 1]] = ((self.tstep * self.dt) + (
                    self.binL - ((self.tstep * self.dt) % self.binL)))
            self.count[spk_idx[:, 0], spk_idx[:, 1]] = self.count[spk_idx[:, 0], spk_idx[:, 1]] + 1
            self.spk_time[spk_idx[:, 0], spk_idx[:, 1]] = self.tstep * self.dt
            self.spike[spk_idx[:, 0], spk_idx[:, 1]] = 1


