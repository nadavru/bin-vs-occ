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


def apply(l, f):
    for x in l:
        f(x)

#######################################
############# parameters ##############
#######################################

# data
d = 20
total_rows = 10000
outliers_ratio = 0.002
test_inliers_ratio = 0.5
test_outliers_ratio = 0.5
data_func = create_data.synth_3
data_name = data_func.__name__

# nn
device = "cuda"
num_epochs = 1 #10

# svm
gamma = "auto"

# test statistic
aggregate_function = "avg"
n_iterations = 200

# P-values
k_times = 1 #50

# graph
alpha = [0.01, 0.15]
n_samples = 100
frac = 0.30 # 30% smoothness

# general
silent = True
data_folder = "data"
results_folder = "results"

new_data = True

#######################################
############# create data #############
#######################################
print("\n" + "#"*20 + " data " + "#"*20 + "\n")

outliers_length = int(outliers_ratio*total_rows)

create_new_data = True
if not new_data:
    if os.path.isfile(f"{data_folder}/{data_name}/in.npy") and os.path.isfile(f"{data_folder}/{data_name}/out.npy"):
        in_dataset = np.load(f"{data_folder}/{data_name}/in.npy")
        out_dataset = np.load(f"{data_folder}/{data_name}/out.npy")
        if in_dataset.shape[0]!=total_rows or out_dataset.shape[0]!=outliers_length or in_dataset.shape[1]!=d:
            print("Found existing data which doesn't fit the parameters. Creates new.")
        else:
            print("Found existing data, using it.")
            create_new_data = False

    else:
        print("Couldn't find existing data. Creates new.")

else:
    print("Creates new data.")

if create_new_data:
    if not os.path.exists(f"{data_folder}/{data_name}"):
        os.makedirs(f"{data_folder}/{data_name}")
    in_dataset = data_func(total_rows, d, in_dist=True)
    out_dataset = data_func(outliers_length, d, in_dist=False)
    np.save(f"{data_folder}/{data_name}/in.npy", in_dataset)
    np.save(f"{data_folder}/{data_name}/out.npy", out_dataset)

#######################################
################ init #################
#######################################
print("\n" + "#"*20 + " init " + "#"*20 + "\n")

# occ
train_inliers_length = int((1-test_inliers_ratio)*total_rows)

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

print("done.")

print("\n" + "#"*20 + " repeat " + "#"*20 + "\n")

for _ in tqdm(range(k_times)):

    #######################################
    ############### permute ###############
    #######################################
    
    # TODO random permutation? new random data? bootstrap of bigger dataset?
    '''in_dataset = np.random.permutation(in_dataset)
    out_dataset = np.random.permutation(out_dataset)'''
    in_dataset = data_func(total_rows, d, in_dist=True)
    out_dataset = data_func(outliers_length, d, in_dist=False)

    apply(t_occ, lambda t: t.reset())
    apply(t_bin, lambda t: t.reset())

    #######################################
    ################ train ################
    #######################################
    
    # occ
    apply(t_occ, lambda t: t.train(in_dataset[:train_inliers_length]))


    # bin
    bin_train_dataset_X = np.concatenate((in_dataset[:train_inliers_length], out_dataset[:train_outliers_length]))
    bin_train_dataset_Y = np.concatenate((np.zeros((train_inliers_length)), np.ones((train_outliers_length))))

    apply(t_bin, lambda t: t.train((bin_train_dataset_X, bin_train_dataset_Y)))

    #######################################
    ########### two sample test ###########
    #######################################

    # occ
    for i, t in enumerate(t_occ):
        occ_p_values[i].append(t.two_sample_test(in_dataset[train_inliers_length:], out_dataset))


    # bin
    for i, t in enumerate(t_bin):
        bin_p_values[i].append(t.two_sample_test(in_dataset[train_inliers_length:], out_dataset[train_outliers_length:]))

#######################################
################ print ################
#######################################

result_folder = f"{results_folder}/{data_name}__{outliers_ratio}"
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
  
x = np.linspace(*alpha, num=n_samples).reshape(-1,1)
all_colors = ["b", "g", "r", "c", "m", "y"]
occ_colors = all_colors[:3]
bin_colors = all_colors[3:]

occ_smoothed = []
bin_smoothed = []

### regular ###

# occ
for p_values, color, name in zip(occ_p_values, occ_colors, occ_names):
    p_values = np.array(p_values).reshape(1,-1)
    powers = (p_values<=x).sum(1)/k_times
    plt.plot(x, powers, color=color, label=name)
    occ_smoothed.append(sm.nonparametric.lowess(powers.squeeze(), x.squeeze(), frac = frac)[:,1])


# bin
for p_values, color, name in zip(bin_p_values, bin_colors, bin_names):
    p_values = np.array(p_values).reshape(1,-1)
    powers = (p_values<=x).sum(1)/k_times
    plt.plot(x, powers, color=color, label=name)
    bin_smoothed.append(sm.nonparametric.lowess(powers.squeeze(), x.squeeze(), frac = frac)[:,1])

plt.xlabel("Alpha")
plt.ylabel("Power")
plt.title("Power of T-statistic")
plt.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(f"{result_folder}/power.png", bbox_inches='tight')
#plt.show()

### smoothed ###

# occ
for powers, color, name in zip(occ_smoothed, occ_colors, occ_names):
    plt.plot(x, powers, color=color, label=name)


# bin
for powers, color, name in zip(bin_smoothed, bin_colors, bin_names):
    plt.plot(x, powers, color=color, label=name)

plt.xlabel("Alpha")
plt.ylabel("Power")
plt.title("Power of T-statistic (smoothed)")
plt.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(f"{result_folder}/smoothed_power.png", bbox_inches='tight')
#plt.show()
