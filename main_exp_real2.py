import occ_nn.model
import occ_svm.model
import occ_forest.model
import bin_svm.model
import bin_forest.model
import bin_nn.model
from test_statistic import TestStatistic
import create_data
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import product
import sys
import seaborn as sn
import math


def apply(l, f):
    for x in l:
        f(x)

#######################################
############# parameters ##############
#######################################

# data
test_inliers_ratio = 0.5
test_outliers_ratio = 0.5

# nn
device = "cuda"
num_epochs = 10

# svm
gamma = "auto"

# test statistic
aggregate_function = "avg"
n_iterations = 200

# P-values
k_times = 100

# graph
alpha = [0.01, 0.15]
n_samples = 100
frac = 0.30 # 30% smoothness
pad = 0.01 # to avoid overlapping

# general
silent = True
data_folder = "data"
results_folder = "results"


datasets = ["breastw", "pima", "satellite", "arrhythmia"]
outliers_ratios = [0.01, 0.04]

all_options = list(product(datasets, outliers_ratios))

if len(sys.argv)>=2:
    all_options = [all_options[int(sys.argv[1])]]
    if len(sys.argv)==3:
        device = f"cuda:{int(sys.argv[2])}"
print(f"using device: {device}")

for (dataset, outliers_ratio) in all_options:

    #######################################
    ############# create data #############
    #######################################

    in_dataset = create_data.real(data_folder, dataset, in_dist=True)
    out_dataset = create_data.real(data_folder, dataset, in_dist=False)

    inliers_length = in_dataset.shape[0]
    d = in_dataset.shape[1]
    total_outliers = out_dataset.shape[0]
    outliers_length = math.ceil(outliers_ratio/(1-outliers_ratio)*inliers_length)

    #######################################
    ################ init #################
    #######################################

    # occ
    train_inliers_length = int((1-test_inliers_ratio)*inliers_length)

    occ_models = [
        occ_nn.model.Model(d=d, num_epochs=num_epochs, device=device, silent=silent), 
        occ_svm.model.Model(gamma=gamma, silent=silent), 
        occ_forest.model.Model(silent=silent)
    ]
    occ_names = ["OCC NN", "OCC SVM", "OCC Isolation Forest"]

    t_occ = [TestStatistic(model, aggregate_function=aggregate_function, n_iterations=n_iterations) for model in occ_models]


    # bin
    train_outliers_length = int((1-test_outliers_ratio)*outliers_length)

    bin_models = [
        bin_nn.model.Model(d=d, num_epochs=num_epochs, device=device, silent=silent), 
        bin_svm.model.Model(gamma=gamma, silent=silent), 
        bin_forest.model.Model(silent=silent)
    ]
    bin_names = ["BIN NN", "BIN SVM", "BIN Random Forest"]

    t_bin = [TestStatistic(model, aggregate_function=aggregate_function, n_iterations=n_iterations) for model in bin_models]

    occ_p_values = [[] for _ in range(3)]
    bin_p_values = [[] for _ in range(3)]

    for _ in tqdm(range(k_times)):

        #######################################
        ############### permute ###############
        #######################################
        
        # TODO random permutation? new random data? bootstrap of bigger dataset?
        in_dataset = np.random.permutation(in_dataset)
        '''indices = np.random.randint(total_outliers, size=outliers_length)
        bootstrapped_out_dataset = out_dataset[indices, :]'''
        train_outliers_indices = np.random.randint(total_outliers, size=train_outliers_length)
        train_outliers_indices_set = set(train_outliers_indices)
        test_outliers_indices = []
        for _ in range(outliers_length-train_outliers_length):
            ind = np.random.randint(total_outliers, size=1)[0]
            while ind in train_outliers_indices_set:
                ind = np.random.randint(total_outliers, size=1)[0]
            test_outliers_indices.append(ind)
        test_outliers_indices = np.array(test_outliers_indices)
        indices = np.concatenate([train_outliers_indices, test_outliers_indices])
        bootstrapped_out_dataset = out_dataset[indices, :]

        apply(t_occ, lambda t: t.reset())
        apply(t_bin, lambda t: t.reset())

        #######################################
        ################ train ################
        #######################################
        
        # occ
        apply(t_occ, lambda t: t.train(in_dataset[:train_inliers_length]))


        # bin
        bin_train_dataset_X = np.concatenate((in_dataset[:train_inliers_length], bootstrapped_out_dataset[:train_outliers_length]))
        bin_train_dataset_Y = np.concatenate((np.zeros((train_inliers_length)), np.ones((train_outliers_length))))

        apply(t_bin, lambda t: t.train((bin_train_dataset_X, bin_train_dataset_Y)))

        #######################################
        ########### two sample test ###########
        #######################################

        # occ
        for i, t in enumerate(t_occ):
            occ_p_values[i].append(t.two_sample_test(in_dataset[train_inliers_length:], bootstrapped_out_dataset))


        # bin
        for i, t in enumerate(t_bin):
            bin_p_values[i].append(t.two_sample_test(in_dataset[train_inliers_length:], bootstrapped_out_dataset[train_outliers_length:]))

    #######################################
    ################ print ################
    #######################################

    result_folder = f"{results_folder}/{dataset}/{outliers_ratio}"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open(f"{result_folder}/p_values.txt", "w") as f:

        f.write("#"*20 + " Mean P-values " + "#"*20 + "\n\n")
        f.write("name: avg | (min , max)\n\n")

        # occ
        for p_values, name in zip(occ_p_values, occ_names):
            mean_p_value = sum(p_values)/k_times
            f.write(f"{name}: {mean_p_value} | ({min(p_values)} , {max(p_values)})\n")

        f.write("\n")

        # bin
        for p_values, name in zip(bin_p_values, bin_names):
            mean_p_value = sum(p_values)/k_times
            f.write(f"{name}: {mean_p_value} | ({min(p_values)} , {max(p_values)})\n")

    #######################################
    ################ graph ################
    #######################################

    outliers_perc = outliers_ratio*100
    if outliers_perc.is_integer():
        outliers_perc = int(outliers_perc)
    else:
        outliers_perc = int(outliers_perc*100)/100
    graph_title = f"{dataset}: {outliers_length} ({outliers_perc}%)"
    
    x = np.linspace(*alpha, num=n_samples).reshape(-1,1)
    all_colors = ["b", "g", "r", "c", "m", "y"]
    occ_colors = all_colors[:3]
    bin_colors = all_colors[3:]

    occ_smoothed = []
    bin_smoothed = []

    ### regular ###

    count = 0
    # occ
    for p_values, color, name in zip(occ_p_values, occ_colors, occ_names):
        p_values = np.array(p_values).reshape(1,-1)
        powers = ((p_values<=x).sum(1)/k_times+count*pad)/(1+5*pad)
        plt.plot(x, powers, color=color, label=name)
        occ_smoothed.append(sm.nonparametric.lowess(powers.squeeze(), x.squeeze(), frac = frac)[:,1])
        count += 1


    # bin
    for p_values, color, name in zip(bin_p_values, bin_colors, bin_names):
        p_values = np.array(p_values).reshape(1,-1)
        powers = ((p_values<=x).sum(1)/k_times+count*pad)/(1+5*pad)
        plt.plot(x, powers, color=color, label=name)
        bin_smoothed.append(sm.nonparametric.lowess(powers.squeeze(), x.squeeze(), frac = frac)[:,1])
        count += 1

    plt.xlabel("Alpha")
    plt.ylabel("Power")
    plt.title(graph_title)
    plt.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([-0.05, 1.05])
    plt.savefig(f"{result_folder}/power.png", bbox_inches='tight')
    plt.close()

    ### smoothed ###

    # occ
    for powers, color, name in zip(occ_smoothed, occ_colors, occ_names):
        plt.plot(x, powers, color=color, label=name)


    # bin
    for powers, color, name in zip(bin_smoothed, bin_colors, bin_names):
        plt.plot(x, powers, color=color, label=name)

    plt.xlabel("Alpha")
    plt.ylabel("Power")
    plt.title(graph_title)
    plt.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([-0.05, 1.05])
    plt.savefig(f"{result_folder}/smoothed_power.png", bbox_inches='tight')
    plt.close()

    cov_in_dataset = (np.corrcoef(in_dataset.T)*10).astype(int)/10
    cov_out_dataset = (np.corrcoef(out_dataset.T)*10).astype(int)/10

    # To address constant features:
    for matrix, d in zip([cov_in_dataset, cov_out_dataset], [in_dataset, out_dataset]):

        diffs = d.max(0) - d.min(0)
        list_inds = []
        for i, diff in enumerate(diffs):
            if diff==0:
                list_inds.append(i)

        for ind in list_inds:
            matrix[ind] = 0.0
            matrix[:,ind] = 0.0
            matrix[ind, ind] = 1.0

    fig, axs = plt.subplots(1,2)
    #fig.subplots_adjust(right=0.8)

    axs[0].axis("square")
    plt.subplot(1,2,1)
    sn.heatmap(cov_in_dataset, annot=False, fmt='g', vmin=-1, vmax=1, cbar=False)
    plt.title(f"Inliers")
    #plt.ylim(-1.05, 1.05)
    plt.subplot(1,2,2)
    axs[1].axis("square")
    sn.heatmap(cov_out_dataset, annot=False, fmt='g', vmin=-1, vmax=1, cbar=False)
    plt.title(f"Outliers")
    #plt.ylim(-1.05, 1.05)
    plt.suptitle(f"{dataset}")
    plt.savefig(f"{result_folder}/correlation_matrix.png", bbox_inches='tight')
    plt.close()
