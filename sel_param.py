"""

Project: EPL(Olfactory bulb) for drift compensation
Script for parameter optimisation (uses grid search)

Author: Ayon Borthakur <ab2535@cornell.edu>
Copyright (c) 2019 Cornell University

"""
# import numpy as np
import os
import datetime
import pickle
from run_parallel import Main
# import torch.multiprocessing as mp
from multiprocessing import Pool
# from genfig import Image


class Optim(object):
    """ Defines the parameter space etc for the model. """

    def __init__(self, nshots=1, ngroups=6, instance_idx=1):
        self.currdir = os.getcwd()
        self.param_space = []
        self.sim_list = []

        self.par = {"num_apimi": 16, "r_apimi": 10., "c_apimi": 2., "t_ref_apimi": 15.,
                    "v_th_apimi": 0.2, "num_et": 16, "r_et": 10., "c_et": 1., "t_ref_et": 15.,
                    "v_th_et": 0.2, "num_pg": 16, "r_pg": 10., "c_pg": 2., "t_ref_pg": 15.,
                    "v_th_pg": 0.2, "num_mi": 16, "mi_dup": 1, "r_mi": 10., "c_mi": 2., "t_ref_mi": 15.,
                    "v_th_mi": 0.4, "r_gr": 5., "c_gr": 0.01, "t_ref_gr": 15., "v_th_gr": 0.35,
                    "syn_wght_etpg": 30., "v_spike": 0.5, "f": 40, "T": 500., "dt": 0.01,
                    "a_p_mg": 0.04, "a_m_mg": 0.02, "tau_m_mg": 5.,
                    "tau_p_mg": 5., "mw_pc": 1.,
                    "tau_m_pc": 2., "a_m_pc": 0.5,
                    "tau2_exc": 0.05, "tau1_exc": 1., "en_exc": 7.,
                    "gmax_exc": 25., "syn_wght": 12.,
                    "cp": 40, "amp_osc": 3.5,
                    "offset_osc": 5., "gamma_osc": 1, "w_scale": 50.,
                    "r": 10, "batchsize": 1, "neurogenesis": False,
                    "display": False, "savefigs": True, "savedata": True}

        path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        # path = os.path.normpath(path + os.sep + os.pardir)
        # path = os.path.normpath(path + os.sep + os.pardir)
        # path = os.path.expanduser('~')
        # instance_idx = 0
        self.data_path = os.path.join(path, 'test/drift/' + str(nshots))
        self.param_path = os.path.join(path, 'test/drift/' + str(nshots))
        self.odor_list = [ 'batch1/set1']
        self.par.update({"odor_list": self.odor_list, "nshots": nshots,
                         "instance_idx": instance_idx, "batch_idx": 1, "ngroups": ngroups})
        self.odor_idx_list = [idx for idx, val in enumerate(self.odor_list)]

    def create_ins(self):

        """ Parameter space. tau_gr >> 0.01 (dt).
        Fall_width max. 80 for rise_width.
        For gr: 1000 , here near_gr: 10. """
        (r_et, v_th_gr, c_gr, r_gr, en_exc, a_p_mg, a_m_pc, tau_p_mg, tau_m_pc,
         v_th_mi_min, v_th_mi_max, v_th_gr_min, v_th_gr_max, num_mi, mi_dup, tau1_exc, tau2_exc, amp_osc, mL,
         neurogenesis, offset_osc, cp, ins, num_gr, mw_pc, syn_wght, dur) = (
             [8], [0.25], [50 * (10**2)], [1. * (10**(-5))],
             [70.], [3.], [50.], [3.], [5.], [0.8], [1.8], [0.3], [0.6], [16], [5],
             [0.5], [1.5], [3.8], [1], [0], [5.], [20], [0], [3200], [1.5], [18.], [25.])

        self.param_space = ["v_th_gr", "c_gr", "r_gr", "en_exc", "num_gr",
                            "a_p_mg", "a_m_pc", "tau_p_mg", "tau_m_pc",
                            "v_th_mi", "tau1_exc", "tau2_exc", "amp_osc",
                            "dur", "offset_osc", "cp", "ins",
                            "mw_pc", "mL", "neurogenesis", "inh", "batchsize"]

        self.sim_list = [Main(**self.par) for self.par["r_et"] in r_et
                         for self.par["v_th_gr"] in v_th_gr
                         for self.par["c_gr"] in c_gr
                         for self.par["r_gr"] in r_gr
                         for self.par["en_exc"] in en_exc
                         for self.par["a_p_mg"] in a_p_mg
                         for self.par["a_m_pc"] in a_m_pc
                         for self.par["tau_p_mg"] in tau_p_mg
                         for self.par["tau_m_pc"] in tau_m_pc
                         for self.par["v_th_mi_min"] in v_th_mi_min
                         for self.par["v_th_mi_max"] in v_th_mi_max
                         for self.par["v_th_gr_min"] in v_th_gr_min
                         for self.par["v_th_gr_max"] in v_th_gr_max
                         for self.par["num_mi"] in num_mi
                         for self.par["mi_dup"] in mi_dup
                         for self.par["tau1_exc"] in tau1_exc
                         for self.par["tau2_exc"] in tau2_exc
                         for self.par["amp_osc"] in amp_osc
                         for self.par["dur"] in dur
                         for self.par["offset_osc"] in offset_osc
                         for self.par["mL"] in mL
                         for self.par["neurogenesis"] in neurogenesis
                         for self.par["cp"] in cp
                         for self.par["ins"] in ins
                         for self.par["num_gr"] in num_gr
                         for self.par["syn_wght"] in syn_wght
                         for self.par["mw_pc"] in mw_pc
                         for self.par["odor_idx_no"] in self.odor_idx_list]

    def multiprocess(self):

        # pool = Pool(processes=2)
        # pool.map(self.parl, self.sim_list)

        for sim_obj in self.sim_list:
            self.parl(sim_obj)

    def parl(self, idx):

        idx.main_loop(self.param_path, self.data_path, self.param_space, self.currdir,
                      self.sim_list.index(idx), self.odor_list)


if __name__ == '__main__':
    ins = Optim()
    ins.create_ins()
    ins.multiprocess()
    '''
    fig_ins = Image(ins.allsimdir)
    fig_ins.create_ins()
    # print(ins.allsimdir)
    fig_ins.multiprocess()
    '''