"""
..Project: EPL(Olfactory bulb)
  Platform: Linux
  Description: Synapse models in PyTorch

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


class Synapse(object):
    """Generic Synapse class. """

    def __init__(self):
        pass

class AMPA1(Synapse):
    """Excitatory AMPA1 synapse without plasticity. """

    def __init__(self):
       pass

class GABA(Synapse):
    """Inhibitory GABA synapse without plasticity. """

    def __init__(self):
       pass


class AMPAstdp(AMPA1):
    """An AMPA STDP synapse
    --------------------------------------
    STDP model 3 from Sjostrom et al 2001.
    """

    def __init__(self, migrsyn_conn_info, prev_syn_wght, w_lim, g_max, **kwargs):

        self.p = kwargs  # for debugging only
        self.gmax_exc = g_max
        self.tau1_exc = kwargs["tau1_exc"]
        self.tau2_exc = kwargs["tau2_exc"]
        self.dt = kwargs["dt"]
        self.a_p = kwargs["a_p_mg"]
        self.a_m = kwargs["a_m_mg"]
        self.tau_p = kwargs["tau_p_mg"]
        self.tau_m = kwargs["tau_m_mg"]
        self.w_max = w_lim
        self.w_scale = kwargs["w_scale"]
        self.binL = kwargs["binL"]
        self.stat = kwargs["stat"]
        self.num_mi = kwargs["num_mi"]
        self.num_gr = kwargs["num_gr"]
        self.batchsize = self.p["batchsize"]
        self.bin_time = 0.
        self.tstep = 0.
        self.migrsyn_conn_info = migrsyn_conn_info
        '''
        if self.stat == 'L':
            active_idx = kwargs["gr_orth_idx"][kwargs["y_label"]]
            self.nonactive_idx = t.LongTensor(np.setdiff1d(
                np.arange(self.num_gr), active_idx))
        '''

        # if iscuda:
        self.g_syn = t.FloatTensor(self.batchsize, self.num_gr).fill_(0)
        self.syn_wght = prev_syn_wght[:, :]
        '''
        self.last_post_spkincyc = -(
            t.FloatTensor(self.num_gr).type(t.FloatTensor))
        '''
        self.last_pre_spk = t.FloatTensor(self.num_mi).fill_(1)
        self.last_pre_spk_nl = t.FloatTensor(self.batchsize, self.num_mi).fill_(1)
        self.last_post_spk = t.FloatTensor(self.num_gr).fill_(1)
        self.last_pre_spk[:] = -10
        self.last_pre_spk_nl[:, :] = -10
        self.last_post_spk[:] = -10
        # self.spk_time = t.FloatTensor(self.num_mi).type(t.FloatTensor)

    def syn_cond(self, pre_spk, post_spk, tstep):

        diff = t.FloatTensor(self.batchsize, self.num_mi, self.num_gr).fill_(0)
        dummy_tensor = t.FloatTensor(self.num_mi, self.num_gr).fill_(0)

        self.tstep = tstep

        if not ((self.tstep * self.dt) % self.binL):
            self.bin_time = self.tstep * self.dt

        pre_spk_nl = pre_spk

        if self.stat == 'L':  # keep learning away from batch - for safety

            pre_spk = torch.squeeze(pre_spk)
            post_spk = torch.squeeze(post_spk)

            self.comp_w_stdp(pre_spk, post_spk)
            '''
            t_diff = ((self.tstep * self.dt) - self.binL)
            if not ((self.tstep * self.dt) % self.binL):
            '''
            if (self.tstep * self.dt) == self.binL:
                # import pdb;pdb.set_trace()
                # pre_idx = (self.last_pre_spk[:] < t_diff).nonzero()
                pre_idx = (self.last_pre_spk[:] < 0.).nonzero()
                if pre_idx.numel():
                    pre_idx = torch.squeeze(pre_idx, 1)
                    dummy_tensor[:, :] = 0.
                    dummy_tensor[pre_idx, :] = dummy_tensor[pre_idx, :] + 1
                    # post_idx = (self.last_post_spk[:] > t_diff).nonzero()
                    post_idx = (self.last_post_spk[:] > 0.).nonzero()
                    if post_idx.numel():
                        post_idx = torch.squeeze(post_idx, 1)
                        dummy_tensor[:, post_idx] = dummy_tensor[:, post_idx] + 1
                        idx = (dummy_tensor[:, :] > 1).nonzero()
                        idx = idx.transpose(1, 0)
                        self.syn_wght[idx[0], idx[1]] = self.syn_wght[idx[0], idx[1]] - self.w_scale
                        '''
                        self.syn_wght[:, :] = (torch.clamp(self.syn_wght[:, :],
                                                           min=0,
                                                           max=self.w_max))
                        '''
                        idx = (self.syn_wght[:, :] < 0.).nonzero()
                        if idx.numel():
                            idx = torch.squeeze(idx, 1)
                            idx = idx.transpose(1, 0)
                            self.syn_wght[idx[0], idx[1]] = 0.
                        idx = (self.syn_wght[:, :] > self.w_max[:, :]).nonzero()
                        if idx.numel():
                            idx = torch.squeeze(idx, 1)
                            idx = idx.transpose(1, 0)
                            self.syn_wght[idx[0], idx[1]] = self.w_max[idx[0], idx[1]]
                        self.syn_wght[:, :] = self.syn_wght[:, :] * self.migrsyn_conn_info

        pre_idx = torch.nonzero(pre_spk_nl[:, :] > 0)
        if pre_idx.numel():
            pre_idx = torch.squeeze(pre_idx, 1)
            self.last_pre_spk_nl[pre_idx[:, 0], pre_idx[:, 1]] = self.tstep * self.dt
            if self.stat == 'L':
                self.last_pre_spk[pre_idx[:, 1]] = self.tstep * self.dt

        pre_idx = torch.nonzero(self.last_pre_spk_nl[:, :] > 0.)
        if pre_idx.numel():
            pre_idx = torch.squeeze(pre_idx, 1)
            '''
            diff[pre_idx[:, 0], pre_idx[:, 1], :] = (torch.reshape((self.dt * self.tstep) - self.last_pre_spk_nl[
                pre_idx[:, 0], pre_idx[:, 1]],(torch.numel(pre_idx), 1))).expand(-1, self.num_gr)
            '''
            diff[pre_idx[:, 0], pre_idx[:, 1], :] = ((self.dt * self.tstep) - self.last_pre_spk_nl[
                pre_idx[:, 0], pre_idx[:, 1]]).unsqueeze_(-1).expand(-1, self.num_gr)
            self.comp_conductance(diff)

    def comp_w_stdp(self, pre_spk, post_spk):
        """
        Based on RIF.
        """
        deltaT = t.FloatTensor(self.num_mi, self.num_gr).fill_(0)
        dummy_tensor = t.FloatTensor(self.num_mi, self.num_gr).fill_(0)
        deltaT[:, :] = 0.

        post_idx = (post_spk[:] > 0).nonzero()
        if post_idx.numel():
            post_idx = torch.squeeze(post_idx, 1)
            dummy_tensor[:, :] = 0.
            dummy_tensor[:, post_idx] = dummy_tensor[:, post_idx] + 1
            pre_idx = (self.last_pre_spk[:] >= self.bin_time).nonzero()
            if pre_idx.numel():
                pre_idx = torch.squeeze(pre_idx, 1)
                dummy_tensor[pre_idx, :] = dummy_tensor[pre_idx, :] + 1
                idx = (dummy_tensor[:, :] > 1).nonzero()
                idx = idx.transpose(1, 0)
                deltaT[idx[0], idx[1]] = (self.tstep * self.dt) - self.last_pre_spk[idx[0]]

                # deltaT[:, :] = deltaT[:, :] * dummy_tensor[:, :]
                deltaT[:, :] = deltaT[:, :] * self.migrsyn_conn_info

                self.last_post_spk[post_idx] = self.tstep * self.dt

        pre_idx = (pre_spk[:] > 0).nonzero()
        if pre_idx.numel():
            pre_idx = torch.squeeze(pre_idx, 1)
            dummy_tensor[:, :] = 0.
            dummy_tensor[pre_idx, :] = dummy_tensor[pre_idx, :] + 1
            post_idx = (self.last_post_spk[:] >= self.bin_time).nonzero()
            if post_idx.numel():
                post_idx = torch.squeeze(post_idx, 1)
                dummy_tensor[:, post_idx] = dummy_tensor[:, post_idx] + 1
                idx = (dummy_tensor[:, :] > 1).nonzero()
                idx = idx.transpose(1, 0)
                deltaT[idx[0], idx[1]] = self.last_post_spk[idx[1]] - (self.tstep * self.dt)

                # deltaT[:, :] = deltaT[:, :] * dummy_tensor[:, :]
                deltaT[:, :] = deltaT[:, :] * self.migrsyn_conn_info

                self.last_pre_spk[pre_idx] = self.tstep * self.dt

        deltaT = deltaT * self.migrsyn_conn_info
        dw = self.comp_dw(deltaT)
        self.scale_w(dw)

    def comp_dw(self, deltaT):
        """
        Computes the weight change as per the exp. STDP rule.
        """
        dw = t.FloatTensor(self.num_mi, self.num_gr).fill_(0)

        idx = (deltaT > 0.).nonzero()
        if idx.numel():
            idx = torch.squeeze(idx, 1)
            idx = idx.transpose(1, 0)
            # import pdb; pdb.set_trace()
            dw[idx[0], idx[1]] = (
                    self.a_p * torch.exp(- deltaT[idx[0], idx[1]] / self.tau_p))
        
        idx = (deltaT < 0.).nonzero()
        if idx.numel():
            idx = torch.squeeze(idx, 1)
            idx = idx.transpose(1, 0)
            dw[idx[0], idx[1]] = (
                    (- self.a_m) * torch.exp(deltaT[idx[0], idx[1]] / self.tau_m))
        dw = dw * self.migrsyn_conn_info
        return dw

    def scale_w(self, dw):

        # weight update
        self.syn_wght[:, :] += dw[:, :]
        idx = (self.syn_wght[:, :] < 0.).nonzero()
        if idx.numel():
            idx = torch.squeeze(idx, 1)
            idx = idx.transpose(1, 0)
            self.syn_wght[idx[0], idx[1]] = 0.
        idx = (self.syn_wght[:, :] > self.w_max[:, :]).nonzero()
        if idx.numel():
            idx = torch.squeeze(idx, 1)
            idx = idx.transpose(1, 0)
            self.syn_wght[idx[0], idx[1]] = self.w_max[idx[0], idx[1]]
        # self.syn_wght[:, :] = (torch.clamp(self.syn_wght[:, :], min=0, max=self.w_max))
        self.syn_wght[:, :] = self.syn_wght[:, :] * self.migrsyn_conn_info[:, :]

    def comp_conductance(self, diff):

        self.syn_wght[:, :] = self.syn_wght[:, :] * self.migrsyn_conn_info[:, :]
        if not ((self.tstep * self.dt) % self.binL):
            '''
            self.spk_time[:] = torch.fmod(self.last_pre_spk[:], self.binL)
            idx = (self.spk_time[:] < 0.).nonzero()
            if idx.numel():
                idx = torch.squeeze(idx, 1)
                self.spk_time[idx] = self.binL
            self.spk_time[:] = (self.binL - self.spk_time[:])/self.binL
            '''
            self.g_syn[:, :] = 0.
            self.last_pre_spk[:] = -10.
            self.last_pre_spk_nl[:, :] = -10
            self.last_post_spk[:] = -10.
        else:
            '''
            self.g_syn[:, :] = ((self.syn_wght[:, :].expand(self.batchsize, -1, -1) *
                                 self.gmax_exc * ((self.tau1_exc * self.tau2_exc) /(
                            self.tau1_exc - self.tau2_exc))) * (torch.exp((-diff[:, :, :]) / self.tau1_exc) -
                                                                torch.exp((-diff[:, :, :]) /self.tau2_exc))).sum(0)
            '''
            self.g_syn[:, :] = ((self.syn_wght[:, :].expand(self.batchsize, -1, -1) *
                                 self.gmax_exc * ((self.tau1_exc * self.tau2_exc) / (
                            self.tau1_exc - self.tau2_exc))) * (torch.exp((-diff[:, :, :]) / self.tau1_exc) -
                                                                torch.exp((-diff[:, :, :]) / self.tau2_exc))).sum(1)


class GABAshunt(GABA):
    """A GABA synapse
    """

    def __init__(self, **kwargs):

        self.p = kwargs
        self.dt = kwargs["dt"]
        self.num_et = int(kwargs["num_et"] / kwargs["mi_dup"])
        self.r = t.FloatTensor(self.p["r_et"])
        # self.num_et = kwargs["num_et"]
        self.T = kwargs["T"]
        self.tstep = 0.

        # if iscuda:
        self.g_syn = t.FloatTensor(self.p["batchsize"], self.num_et).fill_(0)
        # self.g_prf = t.FloatTensor(self.num_et, int(self.T / self.dt)).type(t.FloatTensor)

    def syn_cond(self, pre_cnt, tstep):

        pre_cnt = pre_cnt * self.r
        pre_cnt = torch.sum(pre_cnt, 1)
        self.tstep = tstep
        for i in range(self.p["batchsize"]):
            self.g_syn[i, :] = pre_cnt[i]
        # self.g_prf[:, self.tstep] = self.g_syn[:]


