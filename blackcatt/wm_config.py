import numpy as np
import torch
import os

# Ray object store for shared memory in simulation mode (optional)
try:
    import ray
    USE_RAY_STORE = True
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
except ImportError:
    USE_RAY_STORE = False

wm_constants = "wm_constants/"

# Training
n_users = 2 # = options.num-supernodes in pyproject.toml
user_fraction = 10 / n_users # = fraction-fit in pyproject.toml
mlr = 0.01 # main task learning rate
mbs = 64 # local minibatch size
max_batches = (50000//n_users)//mbs # 1 local epoch total data//mbs
dataset = "CIFAR10" # "CIFAR100" # "CIFAR10"
n_classes = 10 # 10 for CIFAR10, 100 for CIFAR100
model = "ResNet183x3" # "ResNet183x3" # "VGG16"

# Black box WM
tlr = 0.0001 # trigger learning rate
m = 10 # trigger set size
trigger_type = "random" # "random" # "stealthy" # "unique" # "unique" means each user gets their own unique set of triggers, only implemented for vanilla approach, will crash otherwise
lambda_reg = 0.1 # regularization weight # 0.1
k_cols = 5 # number of emulated collusions # 5 
# Trigger Optimization
t_step_pix = 1 # pixel value step size # 1 
t_step = t_step_pix/255*2 # normalized step size
t_alpha_pix = 64 # total pixel value change budget # 64
t_alpha = t_alpha_pix/255*2 # normalized change budget
t_rounds = 1 # trigger optimization rounds # 1
# Functional Regularization
lambda_cl = 0.0 # weight of functional regularization term # 0.1
dataset_cl = "huggan/wikiart" # "huggan/wikiart" # "zh-plus/tiny-imagenet" # "uoft-cs/cifar100"
reserve_testing_images = False # whether to reserve part of testing images for functional regularization / comparison with those experiments

if n_classes == 10:
  tau =  0.01 # Theorem 5 Skoric12 #  
  clients_tardos_q = np.loadtxt(wm_constants + 'clients_tardos_q_10_k0.5.csv').astype(int)
  p_secret = np.loadtxt(wm_constants + 'p_secret_10_k0.5.csv').astype(float)
if n_classes == 100:
  tau =  0.001 # Scaled because of larger n_classes
  clients_tardos_q = np.loadtxt(wm_constants + 'clients_tardos_q_100_k0.5.csv').astype(int)
  p_secret = np.loadtxt(wm_constants + 'p_secret_100_k0.5.csv').astype(float)
clients_tardos_q = clients_tardos_q[:,:m] # Adjust to trigger size
clients_tardos_q_tensor = torch.from_numpy(clients_tardos_q).long()
p_secret_tensor = torch.from_numpy(p_secret[:m,:]).float()

# Generate output folder
triggers_suffix = ""
if trigger_type == "stealthy":
  triggers_suffix = "B9"
elif trigger_type == "unique":
  triggers_suffix = "UQ"

testing_images_suffix = ""
if reserve_testing_images == True:
  testing_images_suffix = "_RTI"

folder = "FL_models/"+model+"_"+dataset+testing_images_suffix+"_" + str(n_users) + "users_0.5k_" + str(m) + "m" + triggers_suffix + "_" + str(mbs) + "mbs_" + str(mlr) + "mlr_" + str(tlr) + "tlr_" + str(max_batches) + "mbatches_" + str(lambda_reg) + "lambdaregCOL" + str(k_cols) + "_" + str(t_alpha_pix) + "talphaTRIG_"+str(dataset_cl.split("/")[-1])+"CLAvgKL"+str(lambda_cl)+"_" + str(t_rounds) + "troundsBest_test/"
if not os.path.exists(folder):
  os.makedirs(folder)

# Recovering from checkpoint
load_checkpoint_flag = False 
loading_round = 0