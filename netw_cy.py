"""
..Project: FeedbackNet
  Platform: Linux
  Description: Mi, Gr, Pi connection network class in PyTorch

..Author: Ayon Borthakur(ab2535@cornell.edu)
"""
import numpy as np
import pandas as pd
# import torch.nn as nn
# import torch.optim as optim
# import torch
import os
import configparser
# from torch.autograd import Variable
import pickle
from neuron_cy import MitralApiLIF, ETLIF, PGLIF, MitralLIF, GranuleLIF
from syn_cy import AMPAstdp, GABAshunt
parser = configparser.ConfigParser()
import torch
iscuda = torch.cuda.is_available()
iscuda = False
if iscuda:
    import torch.cuda as t
else:
    import torch as t


class Network(object):
    """Neural network class."""

    def __init__(self, **kwargs):

        self.syn_dict = {}
        self.p = kwargs 
        self.stat = self.p["stat"]  #status, train or test (Learning, not learning, before learning)
        self.T = self.p["T"]   # total time of simulation
        self.dt = self.p["dt"]
        self.num_gr = self.p["num_gr"]
        self.num_mi = self.p["num_mi"]
        # self.num_apimi = self.p["num_apimi"]
        self.num_et = self.p["num_et"]
        self.num_pg = self.p["num_pg"]

        self.currdir = os.getcwd()

        tau_mi = self.p["r_mi"] * self.p["c_mi"]
        tau_gr = self.p["r_gr"] * self.p["c_gr"]
        tau_apimi = self.p["r_apimi"] * self.p["c_apimi"]  #apical mitral cell 
        tau_et = self.p["r_et"] * self.p["c_et"]
        tau_pg = self.p["r_pg"] * self.p["c_pg"]
        self.binL = 1000 / self.p["f"]   #bin length , f = frequncy 
        # import pdb;pdb.set_trace()
        cycle_n = np.arange(self.T / self.binL) + 1
        t = np.arange(0., self.T, self.dt)
        shape = int(self.T / self.dt)
        # del_shape = int(self.T / self.binL)
        '''
        if self.p["stat"] == 'L':
            y_label = self.p["y_train"][self.p["n"]]
            self.p.update({"y_label": y_label[0]})
        '''

        self.p.update({"tau_mi": tau_mi, "tau_gr": tau_gr,
                       "tau_apimi": tau_apimi, "tau_et": tau_et,
                       "tau_pg": tau_pg, "binL": self.binL,
                       "cycle_n": cycle_n, "t": t,
                       "shape": shape})

        # if iscuda:
        g_max = torch.zeros(self.num_mi, self.num_gr).fill_(0)
        w_maxmat = torch.zeros(self.num_mi, self.num_gr).fill_(0)
        migrsyn_conn_info = torch.zeros(self.num_mi, self.num_gr).fill_(0)

        prev_syn_wght = torch.zeros(self.num_mi, self.num_gr).fill_(0)

        for i in range(self.num_gr):
            if bool(self.p["gr_post"][i]):
                migrsyn_conn_info[:, i] = (migrsyn_conn_info[:, i].index_fill_(0, torch.as_tensor(self.p["gr_post"][i]), 1))

        # MODIFY this in order to optimize memory - loading weights?
        if (self.p["stat"] == 'L' and self.p["n"] > 0) or (self.p["grp_idx"] > 0):  #n - number of samples learned before 
            os.chdir(self.p["prev_dir"])
            data_tmp = pd.read_csv('syn_wght.csv', header=None)
            data_tmp = data_tmp.values
            data_tmp = torch.as_tensor(data_tmp.tolist())
            # self.prev_num_gr = np.shape(data_tmp)[1]
            for i in range(self.num_gr):
                if i < np.shape(data_tmp)[1]:
                    prev_syn_wght[:, i] = data_tmp[:, i]
                else:
                    prev_syn_wght[:, i] = (prev_syn_wght[:, i].index_fill_(0, torch.as_tensor(self.p["gr_post"][i]), self.p["syn_wght"]))

            weights = np.load("weights.npz")

            data_tmp = weights["g_max"]
            data_tmp = torch.as_tensor(data_tmp)

            for i in range(self.num_gr):
                if i < np.shape(data_tmp)[1]:
                    g_max[:, i] = data_tmp[:, i]
                else:
                    g_max[:, i] = torch.as_tensor(np.random.choice(np.linspace(20, 60, np.shape(data_tmp)[0] * 100), np.shape(data_tmp)[0], replace=False))

            data_tmp = weights["w_max"]
            data_tmp = torch.as_tensor(data_tmp)

            for i in range(self.num_gr):
                if i < np.shape(data_tmp)[1]:
                    w_maxmat[:, i] = data_tmp[:, i]
                else:
                    w_maxmat[:, i] = torch.as_tensor(np.random.choice(np.linspace(self.p["syn_wght"], self.p["syn_wght"] * self.p["mw_pc"], self.num_mi * 1000), self.num_mi, replace=False))

            os.chdir(self.currdir)

        elif self.p["stat"] == 'AL' or self.p["stat"] == 'fbL':
            os.chdir(self.p["model_dir"])
            data_tmp = pd.read_csv('syn_wght.csv', header=None)
            data_tmp = data_tmp.values
            data_tmp = torch.as_tensor(data_tmp.tolist())
            prev_syn_wght = data_tmp

            weights = np.load("weights.npz")
            data_tmp = weights["g_max"]
            data_tmp = torch.as_tensor(data_tmp)
            g_max = data_tmp

            data_tmp = weights["w_max"]
            data_tmp = torch.as_tensor(data_tmp)
            w_maxmat = data_tmp

        elif self.p["grp_idx"] == 0:
            if self.stat == 'B4L' or (self.stat == 'L' and self.p["n"] == 0):
                for i in range(self.num_gr):
                    if bool(self.p["gr_post"][i]):
                        prev_syn_wght[:, i] = (prev_syn_wght[:, i].index_fill_(0, torch.as_tensor(self.p["gr_post"][i]), self.p["syn_wght"]))

                        g_max[:, i] = torch.as_tensor(np.random.choice(np.linspace(20, 60, self.num_mi * 100), self.num_mi, replace=False))

                        w_maxmat[:, i] = torch.as_tensor(np.random.choice(np.linspace(self.p["syn_wght"], self.p["syn_wght"] * self.p["mw_pc"], self.num_mi * 100), self.num_mi, replace=False))

        # Create the instances
        # self.apimi_obj = MitralApiLIF(**self.p)
        
        self.et_obj = ETLIF(**self.p)
        self.pg_obj = PGLIF(**self.p)
        self.mi_obj = MitralLIF(**self.p)
        self.gr_obj = GranuleLIF(**self.p)
        

        self.migr_syn_obj = AMPAstdp(migrsyn_conn_info, prev_syn_wght, w_lim=w_maxmat, g_max=g_max, **self.p)
        # self.etpg_syn_obj = AMPAetpg(**self.p)
        self.pget_syn_obj = GABAshunt(**self.p)
        # self.pget_ce_syn_obj = GABAshuntce(**self.p)
        # self.etach_obj = Achtransform(**self.p)

    def simul(self):

        # if iscuda:
        train_data = torch.as_tensor(self.p["i_mi"])
        # spk_time = torch.as_tensor(self.num_mi)
        # spk_time[:] = 0
        # Variable  threshold
        # apimi_v_th = torch.as_tensor(self.p["apimi_v_th"])
        pg_v_th = torch.as_tensor(self.p["pg_v_th"])
        et_v_th = torch.as_tensor(self.p["et_v_th"])
        gr_v_th = torch.as_tensor(self.p["gr_v_th"])
        mi_v_th = torch.as_tensor(self.p["mi_v_th"])
        for i in range(1):
            self.pg_obj.neu_simul(pg_v_th, train_data, i)
            self.pget_syn_obj.syn_cond(self.pg_obj.v, i)
            self.et_obj.neu_simul(et_v_th, train_data, self.pget_syn_obj.g_syn, i)
            # self.etach_obj.syn_cond(self.et_obj.v)
            # self.pget_ce_syn_obj.syn_cond(self.et_obj.v, i)
            # self.ce_resp = self.et_obj.v/self.pget_ce_syn_obj.g_syn
        # apimi_data = self.etach_obj.v_mod.cpu().numpy()
        apimi_data = self.et_obj.v
        # import pdb;pdb.set_trace()
        # apimi_data = apimi_data

        # apimi_data = torch.as_tensor(np.tile(apimi_data, (self.p["mi_dup"],)))

        if self.stat == 'L':
            # active_idx = self.p["gr_orth_idx"][self.p["y_label"]]
            # nonactive_idx = torch.as_tensor(np.setdiff1d(np.arange(self.num_gr), active_idx))
            nonactive_idx = torch.LongTensor(self.p["idx_used_all"])
            # self.syn_wght_time = np.zeros((self.num_mi, self.num_gr, self.p["shape"] + 1))

        for i in range(0, int(self.T / self.dt) + 1, 1):
            # self.apimi_obj.neu_simul(apimi_v_th, apimi_data, i)
            # self.mi_obj.neu_simul(mi_v_th, self.apimi_obj.spike_val, i)
            self.mi_obj.neu_simul(mi_v_th, apimi_data, i)
            '''
            if self.stat == 'L':
                if nonactive_idx.numel():
                    self.gr_obj.spike[0, nonactive_idx] = 0
            '''
            # self.migr_syn_obj.syn_cond(self.mi_obj.spike, self.gr_obj.spike, i)
            # self.gr_obj.neu_simul(gr_v_th, self.migr_syn_obj.g_syn, i)
            
            # print(i)
            '''
            if self.p["stat"] == 'L':
                self.syn_wght_time[:, :, i] = self.migr_syn_obj.syn_wght.cpu().numpy()
            '''
        #import pdb;pdb.set_trace()

        if self.p["stat"] == 'B4L':
            # import pdb;pdb.set_trace()
            idx = (np.where(self.gr_obj.count.cpu().numpy() > 0)[1]).tolist()
            print(self.p["stat"])
            print(len(idx))

        if self.p["stat"] == 'fbL':
            # import pdb;pdb.set_trace()
            idx = (np.where(self.gr_obj.count.cpu().numpy() > 0)[1]).tolist()
            # idx = np.intersect1d(idx, self.p["gr_orth_idx"][self.p["y_fbL"][self.p["n"]][0]])
            self.p["idx_list"][self.p["y_fbL"][self.p["n"]][0] - 1] = idx
            print(self.p["stat"])
            print(len(idx))

        if self.p["stat"] == 'L':
            # import pdb;pdb.set_trace()
            idx = (np.where(self.mi_obj.count.cpu().numpy() > 0)[1]).tolist()
            str = "Mi spike count: {}"
            print(str.format(len(idx)))
            idx = (np.where(self.gr_obj.count.cpu().numpy() > 0)[1]).tolist()
            # idx = np.intersect1d(idx, self.p["gr_orth_idx"][self.p["y_train"][self.p["n"]][0]])
            # self.p["idx_list_all"][self.p["y_train"][self.p["n"]][0] - 1] = idx
            # self.p["idx_list_all"][self.p["n"]] = idx
            self.p["idx_list_all"][self.p["grp_idx"] * self.p["nshots"] + self.p["n"]] = idx
            self.p["idx_used_all"] = np.union1d(self.p["idx_used_all"], idx)
            print(self.p["stat"])
            print(len(idx), len(self.p["idx_used_all"]))
            # print(self.p["idx_list_all"][self.p["grp_idx"] * self.p["nshots"] + self.p["n"]])

    # might not need this if online streaming 
    def save_results(self):
        '''
        count_apimi = self.apimi_obj.count.cpu().numpy()
        spk_time_apimi = self.apimi_obj.spk_time.cpu().numpy()
        count_mi = self.mi_obj.count.cpu().numpy()
        spk_time_mi = self.mi_obj.spk_time.cpu().numpy()
        count_et = self.et_obj.count.cpu().numpy()
        count_pg = self.pg_obj.count.cpu().numpy()
        '''
        '''
        count_gr = self.gr_obj.count.cpu().numpy()
        # spk_time_gr = self.gr_obj.spk_time.cpu().numpy()
        syn_wght = self.migr_syn_obj.syn_wght.cpu().numpy()
        # v_etach = self.etach_obj.v_mod.cpu().numpy()

        if self.stat == 'AL':
            self.p["stat"] = self.p["stat"] + self.p["test"]

        os.chdir(self.p["mydir"])

        y = pd.DataFrame(syn_wght)
        y.to_csv("syn_wght" + self.p["stat"] + str(self.p["grp_idx"]) + str(self.p["n"]) + ".csv",
                 header=None, index=None)
        if self.p["stat"] == 'L':
            g_max = self.migr_syn_obj.gmax_exc.cpu().numpy()
            w_max = self.migr_syn_obj.w_max.cpu().numpy()
        '''
        '''
        if self.p["stat"] == 'L':
            t_results = {"syn_wght_time": self.syn_wght_time}
            np.savez_compressed('t_results' + self.p["stat"] + str(self.p["grp_idx"]), **t_results)
       
        '''
        os.chdir(self.currdir)
        keys = ["mi_pre", "mi_post", "gr_pre", "gr_post", "i_mi", "t"]
        params = {key: self.p[key] for key in self.p if key not in keys}
        parser['Parameters'] = params
        if self.p["stat"] == 'L':
            with open(os.path.join(self.p["mydir"], 'param.ini'), 'w') as configf:
                parser.write(configf)
            with open(os.path.join(self.p["mydir"], 'param.p'), 'wb') as handle:
                pickle.dump(self.p, handle)

        if self.p["stat"] == 'fbL':
            return self.p["mydir"], self.p["idx_list"]
        if self.p["stat"] == 'L':
            return self.p["mydir"], syn_wght, self.p["idx_list_all"], self.p["idx_used_all"], g_max, w_max
