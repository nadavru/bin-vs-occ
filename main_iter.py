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
num_epochs = 2

# svm
gamma = "auto"

# test statistic
aggregate_function = "avg"
n_iterations = 200

# general
silent = False
data_folder = "data"
models_folder = "saved_models"

save_models = False
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
    out_dataset = data_func(int(total_rows*outliers_ratio), d, in_dist=False)
    np.save(f"{data_folder}/{data_name}/in.npy", in_dataset)
    np.save(f"{data_folder}/{data_name}/out.npy", out_dataset)

#######################################
################ train ################
#######################################
print("\n" + "#"*20 + " train " + "#"*20 + "\n")

# occ
train_inliers_length = int((1-test_inliers_ratio)*total_rows)

occ_nn_model = occ_nn.model.Model(d=d, num_epochs=num_epochs, device=device, silent=silent)
occ_svm_model = occ_svm.model.Model(gamma=gamma, silent=silent)
occ_forest_model = occ_forest.model.Model(silent=silent)

t_occ_nn = TestStatistic(occ_nn_model, aggregate_function=aggregate_function, n_iterations=n_iterations)
t_occ_svm = TestStatistic(occ_svm_model, aggregate_function=aggregate_function, n_iterations=n_iterations)
t_occ_forest = TestStatistic(occ_forest_model, aggregate_function=aggregate_function, n_iterations=n_iterations)

t_occ_nn.train(in_dataset[:train_inliers_length])
t_occ_svm.train(in_dataset[:train_inliers_length])
t_occ_forest.train(in_dataset[:train_inliers_length])


# bin
train_outliers_length = int((1-test_outliers_ratio)*outliers_length)

bin_nn_model = bin_nn.model.Model(d=d, num_epochs=num_epochs, device=device, silent=silent)
bin_svm_model = bin_svm.model.Model(gamma=gamma, silent=silent)
bin_forest_model = bin_forest.model.Model(silent=silent)

t_bin_nn = TestStatistic(bin_nn_model, aggregate_function=aggregate_function, n_iterations=n_iterations)
t_bin_svm = TestStatistic(bin_svm_model, aggregate_function=aggregate_function, n_iterations=n_iterations)
t_bin_forest = TestStatistic(bin_forest_model, aggregate_function=aggregate_function, n_iterations=n_iterations)

bin_train_dataset_X = np.concatenate((in_dataset[:train_inliers_length], out_dataset[:train_outliers_length]))
bin_train_dataset_Y = np.concatenate((np.zeros((train_inliers_length)), np.ones((train_outliers_length))))

t_bin_nn.train((bin_train_dataset_X, bin_train_dataset_Y))
t_bin_svm.train((bin_train_dataset_X, bin_train_dataset_Y))
t_bin_forest.train((bin_train_dataset_X, bin_train_dataset_Y))

#######################################
################ save #################
#######################################
if save_models:
    print("\n" + "#"*20 + " save " + "#"*20 + "\n")

    # occ
    t_occ_nn.save(f"occ_nn/{models_folder}")
    t_occ_svm.save(f"occ_svm/{models_folder}")
    t_occ_forest.save(f"occ_forest/{models_folder}")


    # bin
    t_bin_nn.save(f"bin_nn/{models_folder}")
    t_bin_svm.save(f"bin_svm/{models_folder}")
    t_bin_forest.save(f"bin_forest/{models_folder}")

#######################################
################ test #################
#######################################
print("\n" + "#"*20 + " raw scores " + "#"*20 + "\n")

# occ
score_occ_nn = t_occ_nn(in_dataset[train_inliers_length:], out_dataset)
score_occ_svm = t_occ_svm(in_dataset[train_inliers_length:], out_dataset)
score_occ_forest = t_occ_forest(in_dataset[train_inliers_length:], out_dataset)

print(f"OCC NN: {score_occ_nn}\nOCC SVM: {score_occ_svm}\nOCC Isolation Forest: {score_occ_forest}\n")


# bin
score_bin_nn = t_bin_nn(in_dataset[train_inliers_length:], out_dataset[train_outliers_length:])
score_bin_svm = t_bin_svm(in_dataset[train_inliers_length:], out_dataset[train_outliers_length:])
score_bin_forest = t_bin_forest(in_dataset[train_inliers_length:], out_dataset[train_outliers_length:])

print(f"BIN NN: {score_bin_nn}\nBIN SVM: {score_bin_svm}\nBIN Random Forest: {score_bin_forest}")

#######################################
########### two sample test ###########
#######################################
print("\n" + "#"*20 + " P-values " + "#"*20 + "\n")

# occ
score_occ_nn = t_occ_nn.two_sample_test(in_dataset[train_inliers_length:], out_dataset)
score_occ_svm = t_occ_svm.two_sample_test(in_dataset[train_inliers_length:], out_dataset)
score_occ_forest = t_occ_forest.two_sample_test(in_dataset[train_inliers_length:], out_dataset)

print(f"OCC NN: {score_occ_nn}\nOCC SVM: {score_occ_svm}\nOCC Isolation Forest: {score_occ_forest}\n")


# bin
score_bin_nn = t_bin_nn.two_sample_test(in_dataset[train_inliers_length:], out_dataset[train_outliers_length:])
score_bin_svm = t_bin_svm.two_sample_test(in_dataset[train_inliers_length:], out_dataset[train_outliers_length:])
score_bin_forest = t_bin_forest.two_sample_test(in_dataset[train_inliers_length:], out_dataset[train_outliers_length:])

print(f"BIN NN: {score_bin_nn}\nBIN SVM: {score_bin_svm}\nBIN Random Forest: {score_bin_forest}\n")
