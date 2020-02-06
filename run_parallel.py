"""
..Project: FeedbackNet
  Platform: Linux
  Description: Main class for network simulation (using Multiprocessing)

..Author: Ayon Borthakur(ab2535@cornell.edu)
"""
import numpy as np
import pandas as pd
import os
import datetime
import pickle
import torch
iscuda = torch.cuda.is_available()
iscuda = False
if iscuda:
    import torch.cuda as t
else:
    import torch as t
# import torch
from netw_cy import Network
from collections import OrderedDict
# import torch.multiprocessing as mp
# from param_optim import Poptim


class Main(object):
    """Class to run the simulation. """

    def __init__(self, **par):
        self.par = par
        self.gr_post = dict()

    def main_loop(self, param_path, data_path, param_space, currdir, ins_idx, odor_list):

        self.param_space = param_space
        self.currdir = currdir
        self.param_path = param_path
        self.data_path = data_path

        self.timestamp = (datetime.datetime.now().
                          strftime('%Y-%m-%d_%H-%M-%S.%f'))
        self.allsimdir = os.path.join(
            self.data_path, self.timestamp + '_{}'.format(os.path.basename(__file__)[:-3]))
        os.makedirs(self.allsimdir)
        self.sim_main_dir = self.allsimdir

        dir_name = self.timestamp + "__sim_info"
        sim_info = os.path.join(self.data_path, dir_name)
        os.makedirs(sim_info)
        os.chdir(sim_info)
        with open('p_sp_info.p', 'wb') as handle:
            pickle.dump(self.param_space, handle)
        os.chdir(self.currdir)

        ngroups = self.par["ngroups"]
        group_type = []
        for i in range(1, ngroups+1):
            group_type.append('group' + str(i))

        # group_type = ['group1', 'group2', 'group3', 'group4']
        '''
        os.chdir(self.allsimdir)
        sim_main_dir = os.path.join(os.getcwd(),
                                         datetime.datetime.now().
                                         strftime('%Y-%m-%d_%H-%M-%S.%f')
                                         + '_' + str(ins_idx))
        os.makedirs(sim_main_dir)
        self.sim_main_dir = sim_main_dir
        '''
        self.par["num_et"] = self.par["mi_dup"] * self.par["num_et"]
        self.par["num_mi"] = self.par["mi_dup"] * self.par["num_mi"]
        # self.par["num_apimi"] = self.par["mi_dup"] * self.par["num_apimi"]

        # self.par["syn_wght"] = 12.
        # self.par["gmax_exc"] = 25.
        # self.par["w_max"] = self.par["mw_pc"] * self.par["syn_wght"]
        self.par["tau_m_mg"] = self.par["tau_m_pc"] * self.par["tau_p_mg"]
        self.par["a_m_mg"] = self.par["a_m_pc"] * self.par["a_p_mg"]

        self.par["pg_v_th"] = np.zeros(self.par["num_pg"])
        val_list = np.linspace(0.4, 1.2, self.par["mi_dup"])
        for i, j in enumerate(val_list):
            self.par["pg_v_th"][i * int(
                self.par["num_pg"] / self.par["mi_dup"]):(i + 1) * int(
                self.par["num_pg"] / self.par["mi_dup"])] = j
        self.par["pg_v_th"][:] = 0.4

        self.par["et_v_th"] = np.zeros(self.par["num_et"])
        val_list = np.linspace(0.4, 1.2, self.par["mi_dup"])
        for i, j in enumerate(val_list):
            self.par["et_v_th"][i * int(
                self.par["num_et"] / self.par["mi_dup"]):(i + 1) * int(
                self.par["num_et"] / self.par["mi_dup"])] = j
        self.par["et_v_th"][:] = 0.4
        '''
        # Variable apical mitral cell threshold
        self.par["apimi_v_th"] = np.zeros(self.par["num_apimi"])
        val_list = np.linspace(self.par["v_th_apimi_min"], self.par["v_th_apimi_max"],
                               self.par["mi_dup"])

        for apimi_idx in range(int(self.par["num_apimi"] / self.par["mi_dup"])):
            for i, j in enumerate(val_list):
                self.par["apimi_v_th"][int(apimi_idx * self.par["mi_dup"] + i)] = j
        '''
        # Variable mitral cell threshold
        self.par["mi_v_th"] = np.zeros(self.par["num_mi"])
        val_list = np.linspace(self.par["v_th_mi_min"], self.par["v_th_mi_max"],
                               self.par["mi_dup"])

        for mi_idx in range(int(self.par["num_mi"] / self.par["mi_dup"])):
            shuff_thres = np.random.permutation(val_list)
            for i, j in enumerate(shuff_thres):
                self.par["mi_v_th"][int(mi_idx * self.par["mi_dup"] + i)] = j
        # self.par["mi_v_th"][:] = self.par["v_th_mi"]

        # Variable granule cell threshold
        self.par["gr_v_th"] = np.zeros(self.par["num_gr"])
        val_list = np.linspace(self.par["v_th_gr_min"], self.par["v_th_gr_max"], 10)
        for i, j in enumerate(val_list):
            self.par["gr_v_th"][i * int(self.par["num_gr"] / 10):
                                (i + 1) * int(self.par["num_gr"] / 10)] = j

        self.idx_used_all = []
        # self.idx_list_all = [[[] for _ in range(self.par["nshots"])] for _ in range(len(group_type))]
        self.idx_list_all = [[] for _ in range(len(group_type) * self.par["nshots"])]
        self.par.update({"idx_used_all": self.idx_used_all,
                         "idx_list_all": self.idx_list_all})
        '''
        self.par["r_et"] = np.asarray([0.80445861, 0.5955682 , 0.96356999, 0.93477695, 0.68354941,
                            0.54698896, 0.79151103, 0.95590562, 0.96749636, 0.95124595,
                            0.53562573, 0.7577298 , 0.52956828, 0.74933508, 0.97499955,
                            0.84195266])
        '''
        self.par["r_et"] = np.random.uniform(0.5, 1., int(self.par["num_mi"]/self.par["mi_dup"]))
        # self.par["r_et"] = np.asarray([1.])
        sensor2et = np.zeros((int(self.par["num_et"] / self.par["mi_dup"]), self.par["num_et"]))
        et2api = np.zeros((self.par["num_et"], self.par["num_mi"]))
        for i in range(int(self.par["num_et"] / self.par["mi_dup"])):
            sensor2et[i, (i * self.par["mi_dup"]): (i * self.par["mi_dup"]) + self.par["mi_dup"]] = 1.
            # sensor2et[i, i] = 1.
        for i in range(int(self.par["num_et"] / self.par["mi_dup"])):
            # et2api[i, i] = 1.
            id_list = [k for k in range(i * self.par["mi_dup"], (i * self.par["mi_dup"]) + self.par["mi_dup"])]
            for j in range(self.par["mi_dup"]):
                idx = np.random.choice(id_list, 2, replace=False)
                # et2api[i * 5 + j, idx] = np.linspace(0.4, 2., 2)
                # et2api[i * self.par["mi_dup"] + j, idx] = 1.
                et2api[i * self.par["mi_dup"] + j, idx] = np.random.choice(
                    np.linspace(0.25, 0.75, 1000), 2, replace=False)

        self.par.update({"sensor2et": sensor2et, "et2api": et2api})

        for grp_idx, grp_type in enumerate(group_type):

            # import pdb;pdb.set_trace()
            self.par["odor_id"] = odor_list[self.par["odor_idx_no"]]

            # path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
            # path = os.path.normpath(path + os.sep + os.pardir)
            # path = os.path.normpath(path + os.sep + os.pardir)
            # path = os.path.join(path, 'data')
            self.odor_path = os.path.join(self.currdir, self.par["odor_id"])
            self.odor_path = os.path.join(self.odor_path, grp_type)

            if grp_idx == 0:

                # Variable connection probability
                '''
                """The Mitral &  Granule connection details. """
                mi_pre = self.get_nose(self.par["num_gr"], self.par["num_mi"])
                """Feedforward connection. Random"""
                self.gr_post = {i: list() for i in range(self.par["num_gr"])}
                for key, value in mi_pre.items():
                    for v in value:
                        gr_id = v
                        self.gr_post[gr_id].append(key)

                # self.gr_post = self.get_nose(self.par["num_mi"], self.par["num_gr"])
                '''
                val = np.arange(1, min(int(self.par["num_mi"]/2), 41))
                mi_val = np.arange(self.par["num_mi"])
                self.gr_post = {i: list() for i in range(self.par["num_gr"])}
                for key, value in self.gr_post.items():
                    self.gr_post[key] = np.random.choice(mi_val, np.random.choice(val, 1, replace=False)[0],
                                                         replace=False).tolist()
                self.par.update({"gr_post": self.gr_post, "grp_idx": grp_idx})
                # import pdb;pdb.set_trace()

                os.chdir(self.param_path)
                conn_dict = {"gr_post": self.gr_post}
                with open(os.path.join("conn_dict" + str(self.par["instance_idx"]) + ".p"), 'wb') as handle:
                    pickle.dump(conn_dict, handle)
                os.chdir(currdir)

            else:
                os.chdir(self.param_path)
                with open('conn_dict' + str(self.par["instance_idx"]) + '.p', "rb") as handle:
                    p = pickle.load(handle)
                self.par.update({"gr_post": p["gr_post"], "grp_idx": grp_idx})
                os.chdir(currdir)

            os.chdir(self.odor_path)
            self.train_data = []
            # import pdb;pdb.set_trace()
            self.df = pd.read_csv('train_data_scaled.csv', header=None)
            # import pdb;pdb.set_trace()
            self.df = self.df.values
            # idx = np.random.choice(6, 6 , replace=False)
            self.train_data.append(self.df)
            y_train = pd.read_csv('y_train.csv', header=None)
            self.y_train = y_train.values
            # import pdb;pdb.set_trace()
            self.par.update({"y_train": self.y_train})

            self.fbL_data = []
            self.df = pd.read_csv('fbL_data_scaled.csv', header=None)
            # import pdb;pdb.set_trace()
            self.df = self.df.values
            # idx = np.random.choice(6, 6 , replace=False)
            self.fbL_data.append(self.df)
            y_fbL = pd.read_csv('y_fbL.csv', header=None)
            self.y_fbL = y_fbL.values
            # import pdb;pdb.set_trace()
            self.par.update({"y_fbL": self.y_fbL})

            df = pd.read_csv('fbL_data_scaled.csv', header=None)
            # df = pd.read_csv('train_data_ntce2_b2.csv', header=None)
            df = df.values
            self.val_data = []
            self.val_data.append(df)

            df = pd.read_csv('fbL_data_scaled.csv', header=None)
            df = df.values
            self.test_data = []
            self.test_data.append(df)

            os.chdir(self.currdir)
            self.par.update({"sim_main_dir": self.sim_main_dir, "stat": 'B4L',
                             "odor_path": self.odor_path})
            self.odor_seq_loop()

            os.chdir(self.currdir)

            self.par.update({"sim_main_dir": self.sim_main_dir, "stat": 'L',
                             "odor_path": self.odor_path})
            self.odor_seq_loop()
            os.chdir(self.currdir)

            self.par.update({"sim_main_dir": self.sim_main_dir, "stat": 'AL',
                             "odor_path": self.odor_path, "test": 'train_test'})
            self.odor_seq_loop()

            os.chdir(self.currdir)

            self.par.update({"sim_main_dir": self.sim_main_dir, "stat": 'AL',
                             "odor_path": self.odor_path, "test": 'val_test'})
            self.odor_seq_loop()

            os.chdir(self.currdir)

            self.par.update({"sim_main_dir": self.sim_main_dir, "stat": 'AL',
                             "odor_path": self.odor_path, "test": 'test_test'})
            self.odor_seq_loop()

            os.chdir(self.currdir)

    def odor_seq_loop(self):

        os.chdir(self.currdir)
        if self.par["stat"] == 'L':

            if self.par["grp_idx"] > 0:
                os.chdir(self.sim_main_dir)
                with open('idx_used_all.p', "rb") as handle:
                    idx_used_all = pickle.load(handle)
                with open('idx_list_all.p', "rb") as handle:
                    idx_list_all = pickle.load(handle)
                os.chdir(self.currdir)

                self.par.update({"idx_used_all": idx_used_all,
                                 "idx_list_all": idx_list_all})

            self.par["batchsize"] = 1
            os.chdir(self.currdir)
            i_mi = self.train_data[0][0, :]
            # i_mi = np.tile(i_mi, (self.par["mi_dup"],))
            # import pdb;pdb.set_trace()
            if self.par["grp_idx"] > 0:
                self.par.update({"prev_dir": self.sim_main_dir})

            self.par.update({"n": 0, "mydir": self.sim_main_dir, "i_mi": i_mi})
            self.par["T"] = self.par["dur"]
            mn = Network(**self.par)
            mn.simul()
            #self.prev_dir, syn_wght, idx_list_all, idx_used_all, g_max_ff, w_max_ff = mn.save_results()

            os.chdir(self.sim_main_dir)
            y = pd.DataFrame(syn_wght)
            y.to_csv("syn_wght.csv", header=None, index=None)

            weights = {"g_max": g_max_ff, "w_max": w_max_ff}
            np.savez_compressed('weights', **weights)

            with open(os.path.join("idx_list_all" + ".p"), 'wb') as handle:
                pickle.dump(idx_list_all, handle)
            with open(os.path.join("idx_used_all" + ".p"), 'wb') as handle:
                pickle.dump(idx_used_all, handle)

            del mn
            # t.empty_cache()
            os.chdir(self.currdir)

            for i in range(1, len(self.y_train)):

                os.chdir(self.currdir)
                i_mi = self.train_data[0][i, :]
                self.par.update({"n": i, "mydir": self.sim_main_dir,
                                 "prev_dir": self.prev_dir, "i_mi": i_mi,
                                 "idx_list_all": idx_list_all, "idx_used_all": idx_used_all})
                self.par["T"] = self.par["dur"]
                mn = Network(**self.par)
                mn.simul()
                self.prev_dir, syn_wght, idx_list_all, idx_used_all, g_max_ff, w_max_ff = mn.save_results()

                os.chdir(self.sim_main_dir)
                y = pd.DataFrame(syn_wght)
                y.to_csv("syn_wght.csv", header=None, index=None)

                os.chdir(self.sim_main_dir)
                weights = {"g_max": g_max_ff, "w_max": w_max_ff}
                np.savez_compressed('weights', **weights)

                with open(os.path.join("idx_list_all" + ".p"), 'wb') as handle:
                    pickle.dump(idx_list_all, handle)
                with open(os.path.join("idx_used_all" + ".p"), 'wb') as handle:
                    pickle.dump(idx_used_all, handle)

                del mn
                # t.empty_cache()

            os.chdir(self.currdir)

        elif self.par["stat"] == 'B4L':
            # t.empty_cache()
            pass
        else:

            if self.par["test"] == 'train_test':
                no_test_items = np.size(self.fbL_data[0], 0)
            elif self.par["test"] == 'val_test':
                no_test_items = len(self.val_data[0])
            elif self.par["test"] == 'test_test':
                no_test_items = len(self.test_data[0])

            def_batchsize = 1
            no_batches = max(1, int(no_test_items/def_batchsize) + 1)

            for n in range(0, no_batches):
                os.chdir(self.currdir)
                # import pdb;pdb.set_trace()
                if self.par["test"] == 'train_test':
                    end_batch = min(no_test_items, (n * def_batchsize) + def_batchsize)
                    self.par["batchsize"] = end_batch - (n * def_batchsize)
                    i_mi = self.fbL_data[0][(n * def_batchsize):end_batch, :]
                elif self.par["test"] == 'val_test':
                    end_batch = min(no_test_items, (n * def_batchsize) + def_batchsize)
                    self.par["batchsize"] = end_batch - (n * def_batchsize)
                    i_mi = self.val_data[0][(n * def_batchsize):end_batch, :]
                elif self.par["test"] == 'test_test':
                    end_batch = min(no_test_items, (n * def_batchsize) + def_batchsize)
                    self.par["batchsize"] = end_batch - (n * def_batchsize)
                    i_mi = self.test_data[0][(n * def_batchsize):end_batch, :]

                self.par["T"] = self.par["dur"]
                self.par.update({"n": (n * def_batchsize), "mydir": self.sim_main_dir,
                                 "model_dir": self.sim_main_dir, "i_mi": i_mi, "stat": 'AL'})
                mn = Network(**self.par)
                mn.simul()
                mn.save_results()
                del mn
                # t.empty_cache()
                # print(n)

    def get_nose(self, num, num_cells):

        cp_list = np.linspace(self.par["cp"], self.par["cp"], num_cells)
        cell_idx = np.random.permutation(np.arange(num_cells))

        conn_dict = {i: [] for i in range(num_cells)}

        for idx, cp in enumerate(cp_list):
            conn_dict[cell_idx[idx]] = list(np.random.choice(num, max(int((cp * num) / 100), 1), replace=False))
        return conn_dict