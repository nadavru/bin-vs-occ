import numpy as np
import os
import scipy.io
import mat73

### mean change ###

# iid
def synth_1(total_rows, d, in_dist=True):
    db = np.random.standard_normal((total_rows,d)) # [total_rows,d]
    if not in_dist:
        db += 0.1
    return db

# corr
def synth_2(total_rows, d, in_dist=True):
    db = np.random.standard_normal((total_rows,d)) # [total_rows,d]
    if not in_dist:
        db += 0.1
    for i in range(1,d):
        db[:,i] += db[:,i-1]
    return db

### std change ###

# iid
def synth_3(total_rows, d, in_dist=True):
    if in_dist:
        db = np.random.standard_normal((total_rows,d)) # [total_rows,d]
    else:
        db = np.random.standard_normal((total_rows,d))*1.1 # [total_rows,d]
    return db

# corr
def synth_4(total_rows, d, in_dist=True):
    db = np.random.standard_normal((total_rows,d)) # [total_rows,d]
    if not in_dist:
        db *= 1.1
    for i in range(1,d):
        db[:,i] += db[:,i-1]
    return db

### adversarial example ###
def synth_5(total_rows, d, in_dist=True):
    db = np.random.standard_normal((total_rows,d)) # [total_rows,d]
    if not in_dist:
        db[:,-d//2:] = -db[:,:d//2]
    return db

### real data ###
def real(data_folder, name, in_dist=True):
    assert os.path.isfile(f"{data_folder}/{name}.mat"), f"'{name}.mat' doesn't exists in {data_folder} folder"
    try:
        mat = scipy.io.loadmat(f"{data_folder}/{name}.mat")
    except:
        mat = mat73.loadmat(f"{data_folder}/{name}.mat")
    x, y = mat['X'], mat['y'].squeeze()
    db = x[y!=in_dist] # 0 means in distribution
    return db
