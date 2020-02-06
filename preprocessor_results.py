"""
..Project: NICE abstract
  Platform: Linux
  Description: NTCE implementation

..Author: Ayon Borthakur(ab2535@cornell.edu)
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from init_plot import init_plotting
init_plotting()

batches = [1]

for btc in batches:

    currdir = os.getcwd()
    path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    path = os.path.normpath(path + os.sep + os.pardir)
    path = os.path.normpath(path + os.sep + os.pardir)
    path = os.path.join(path, 'data')
    path = os.path.join(path, 'fneuro_data/4shot/driftdata4e/b' + str(btc) + 'raw/multraw')

    os.chdir(currdir)
    re_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    re_path = os.path.normpath(re_path + os.sep + os.pardir)
    re_path = os.path.normpath(re_path + os.sep + os.pardir)
    re_path = os.path.join(re_path, 'junk')
    os.chdir(path)
    # Data extraction
    df_train = pd.read_csv('train_data.csv', header=None)
    df_train = df_train.values
    df_train[df_train < 0] = 0
    '''
    df_val = pd.read_csv('val_data.csv', header=None)
    df_val = df_val.values
    '''

    y_train = pd.read_csv('y_train.csv', header=None)
    y_train = y_train.values

    # Unit scaling
    # uscaled_array = np.max(df_val, 0)
    # Unit scaling

    os.chdir(re_path)

    uscaled_array = pd.read_csv('uscaled_array.csv', header=None)
    uscaled_array.values
    uscaled_array = np.asarray(uscaled_array).flatten()

    id_0 = pd.read_csv('id_0.csv', header=None)
    id_0.values
    id_0 = np.asarray(id_0).flatten()

    id_het = pd.read_csv('id_het.csv', header=None)
    id_het.values
    id_het = np.asarray(id_het).flatten()

    os.chdir(currdir)
    df1_train = df_train/uscaled_array
    # df1_val = df_val/uscaled_array
    # df1_test = df_test/uscaled_array
    # df2_train = df1_train

    r_et = np.asarray([0.80445861, 0.5955682, 0.96356999, 0.93477695, 0.68354941,
                       0.54698896, 0.79151103, 0.95590562, 0.96749636, 0.95124595,
                       0.53562573, 0.7577298, 0.52956828, 0.74933508, 0.97499955,
                       0.84195266])
    df2_train = df1_train * r_et

    df2_train_sum = np.sum(df2_train, 1)
    df3_train = df2_train/np.transpose(np.tile(df2_train_sum, (np.size(df2_train, 1), 1)))

    # df3_train_sum = np.sum(df3_train, 1)
    # import pdb;pdb.set_trace()
    # df3_train = df3_train / np.transpose(np.tile(df3_train_sum, (np.size(df3_train, 1), 1)))
    # df3_train = df3_train * 10

    sensor2et = np.zeros((16, 80))
    et2api = np.zeros((80, 80))
    for i in range(16):
        sensor2et[i, (i * 5): (i * 5) + 5] = 1.
    for i in range(16):
        id_list = [k for k in range(i * 5, (i * 5) + 5)]
        for j in range(5):
            idx = np.random.choice(id_list, 2, replace=False)
            et2api[i * 5 + j, idx] = np.random.choice(np.linspace(0., 1., 1000), 2, replace=False)
            # et2api[i * 5 + j, idx] = np.linspace(0.5, 2., 2)
            # et2api[i * 5 + j, idx] = 1.
    x = (np.mat(df3_train) * np.mat(sensor2et))
    # x_sum = np.sum(x, 1)
    # import pdb;pdb.set_trace()
    # x = x / x_sum
    x = x * 10
    df4_train = x * np.mat(et2api)
    '''
    df4_train_sum = np.sum(df4_train, 1)
    # import pdb;pdb.set_trace()
    df4_train = df4_train / df4_train_sum
    '''

    df4_train = df4_train * 1.
    # data_train = df4_train
    data_train = df_train
    '''
    id_0 = np.flipud(np.argsort(df_train[0][:]))

    id_het = np.flipud(np.argsort(df4_train[0][:]))
    '''
    color_list = ["g", "r", "b", "k", "y",  "c"]
    label_list = ["Ammonia", "Acetaldehyde", "Acetone", "Ethylene", "Ethanol", "Toluene"]
    ##########################################################
    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.transpose(np.squeeze(np.transpose(data_train[i, :])[id_0])),
                 marker="*", color=color_list[int(y_train[i]-1)])
    # import pdb; pdb.set_trace()
    plt.ylim([0., 7*(10**5)])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 5))
    os.chdir(re_path)
    plt.savefig("raw_train.svg")
    plt.savefig("raw_train.png")
    plt.close("all")
    os.chdir(os.getcwd())

    if btc == 7:
        plt.figure()
        for i in range(len(y_train)):
            if not i % 4:
                plt.plot(np.transpose(np.squeeze(np.transpose(data_train[i, :])[id_0])),
                         label=label_list[int(y_train[int(i)]) - 1], marker="*", color=color_list[int(y_train[i] - 1)])
            else:
                plt.plot(np.transpose(np.squeeze(np.transpose(data_train[i, :])[id_0])),
                         marker="*", color=color_list[int(y_train[i] - 1)])
        # import pdb; pdb.set_trace()
        plt.ylim([0., 7 * (10 ** 5)])
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 5))
        plt.legend()
        os.chdir(re_path)
        plt.savefig("raw_train.svg")
        plt.savefig("raw_train.png")
        plt.close("all")

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.sort(np.transpose(np.squeeze(np.transpose(data_train[i, :]))), 0)[::-1], marker="*", color=color_list[int(y_train[i]-1)])
    # import pdb; pdb.set_trace()
    plt.ylim([0., 7 * (10 ** 5)])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 5))
    plt.xticks([])
    os.chdir(re_path)
    plt.savefig("raw_train_sorted.svg")
    plt.savefig("raw_train_sorted.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.transpose(np.squeeze(np.transpose(df1_train[i, :])[id_0])), marker="*", color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    plt.ylim([0., 1.5])
    plt.xticks([])
    os.chdir(re_path)
    plt.savefig("scaled_train.svg")
    plt.savefig("scaled_train.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.sort(np.transpose(np.squeeze(np.transpose(df1_train[i, :]))), 0)[::-1], marker="*",
                 color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    plt.ylim([0., 1.5])
    # plt.xticks([])
    # plt.yticks([])
    os.chdir(re_path)
    plt.savefig("scaled_train_sorted.svg")
    plt.savefig("scaled_train_sorted.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.transpose(np.squeeze(np.transpose(df2_train[i, :])[id_0])), marker="*", color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    plt.xticks([])
    # plt.yticks([])
    plt.ylim([0., 1.5])
    os.chdir(re_path)
    plt.savefig("uni_train.svg")
    plt.savefig("uni_train.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.sort(np.transpose(np.squeeze(np.transpose(df2_train[i, :]))), 0)[::-1], marker="*",
                 color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    # plt.xticks([])
    # plt.yticks([])
    plt.ylim([0., 1.5])
    plt.xticks([])
    os.chdir(re_path)
    plt.savefig("uni_train_sorted.svg")
    plt.savefig("uni_train_sorted.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.transpose(np.squeeze(np.transpose(df3_train[i, :])[id_0])), marker="*", color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    # plt.xticks([])
    # plt.yticks([])
    plt.ylim([0., 0.2])
    os.chdir(re_path)
    plt.savefig("ntce2_train.svg")
    plt.savefig("ntce2_train.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.sort(np.transpose(np.squeeze(np.transpose(df3_train[i, :]))), 0)[::-1],
                 marker="*", color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    # plt.xticks([])
    # plt.yticks([])
    plt.ylim([0., 0.2])
    plt.xticks([])
    os.chdir(re_path)
    plt.savefig("ntce2_train_sorted.svg")
    plt.savefig("ntce2_train_sorted.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.transpose(np.squeeze(np.transpose(df4_train[i, :])[id_het])), marker="*", color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    # plt.ylim([0., 4.])
    # plt.xticks([])
    # plt.yticks([])
    os.chdir(re_path)
    plt.savefig("het_train.svg")
    plt.savefig("het_train.png")
    plt.close("all")
    os.chdir(os.getcwd())

    plt.figure()
    for i in range(len(y_train)):
        plt.plot(np.sort(np.transpose(np.squeeze(np.transpose(df4_train[i, :]))), 0)[::-1], marker="*",
                 color=color_list[int(y_train[i] - 1)])
    # import pdb; pdb.set_trace()
    # plt.ylim([0., 4.])
    # plt.xticks([])
    # plt.yticks([])
    plt.xticks([])
    os.chdir(re_path)
    plt.savefig("het_train_sorted.svg")
    plt.savefig("het_train_sorted.png")
    plt.close("all")
    os.chdir(os.getcwd())

    '''
    os.chdir(re_path)
    y = pd.DataFrame(np.asarray(id_0))
    y.to_csv("id_0.csv", header=None, index=None)

    os.chdir(re_path)
    y = pd.DataFrame(np.asarray(uscaled_array))
    y.to_csv("uscaled_array.csv", header=None, index=None)
    if btc == 1:
        os.chdir(re_path)
        y = pd.DataFrame(np.asarray(id_het))
        y.to_csv("id_het.csv", header=None, index=None)
    '''
    os.chdir(re_path)
    y = pd.DataFrame(np.asarray(df_train))
    y.to_csv("df_train.csv", header=None, index=None)
    y = pd.DataFrame(np.asarray(df1_train))
    y.to_csv("df1_train.csv", header=None, index=None)
    y = pd.DataFrame(np.asarray(df2_train))
    y.to_csv("df2_train.csv", header=None, index=None)
    y = pd.DataFrame(np.asarray(df3_train))
    y.to_csv("df3_train.csv", header=None, index=None)
    y = pd.DataFrame(np.asarray(df4_train))
    y.to_csv("df4_train.csv", header=None, index=None)
    os.chdir(currdir)