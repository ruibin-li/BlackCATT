from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import ParametersRecord, array_from_numpy
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage, Lambda, Resize, CenterCrop
import blackcatt.wm_config as wm_config
import blackcatt.models as models
from blackcatt.task import set_weights, get_weights, test, fds, load_state_from_record
import numpy as np
import pickle
import shutil
import os
from PIL import Image
import torch.nn.utils.prune as prune
import os

## Constants and Caches ##
fds_aux = None  # Cache FederatedDataset for functional regularization
# Pre-create trigger transforms to avoid repeated Compose creation
TRIGGER_TRANSFORMS = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# Ray object store for simulation mode - primary storage for client states
_ray_client_states = {}  # dict mapping cid -> ray.ObjectRef

## Functional Regularization specific functions ##
def load_aux_data(aux_cid=0, batch_size=wm_config.mbs):

    global fds_aux

    if wm_config.dataset_cl == "zh-plus/tiny-imagenet":
        # 500 samples per partition
        partitioner_aux = IidPartitioner(num_partitions=200)
        if fds_aux is None:
            fds_aux = FederatedDataset(
                dataset="zh-plus/tiny-imagenet",
                partitioners={"train": partitioner_aux},
            )
        partition_aux = fds_aux.load_partition(aux_cid)
        pytorch_transforms_aux = Compose(
            [ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            Resize((32, 32))]  # Resize to match model input size
        )
        def apply_transforms_aux(batch):
            batch["image"] = [pytorch_transforms_aux(img) for img in batch["image"]]
            return batch

        testset_aux = partition_aux.with_transform(apply_transforms_aux)
        testloader_aux = DataLoader(testset_aux, batch_size=batch_size, shuffle=False)

    elif wm_config.dataset_cl == "huggan/wikiart":
        # 500 samples per partition
        partitioner_aux = IidPartitioner(num_partitions=163)
        if fds_aux is None:
            fds_aux = FederatedDataset(
                dataset="huggan/wikiart",
                partitioners={"train": partitioner_aux},
            )
        partition_aux = fds_aux.load_partition(aux_cid)
        pytorch_transforms_aux = Compose(
            [ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            CenterCrop((512, 512)),
            Resize((32, 32))]  # Crop and Resize to match model input size
        )
        def apply_transforms_aux(batch):
            batch["image"] = [pytorch_transforms_aux(img) for img in batch["image"]]
            return batch

        testset_aux = partition_aux.with_transform(apply_transforms_aux)
        testloader_aux = DataLoader(testset_aux, batch_size=batch_size, shuffle=False)
    
    elif wm_config.dataset_cl == "uoft-cs/cifar100":
        partitioner_aux = IidPartitioner(num_partitions=wm_config.n_users)
        if fds_aux is None:
            fds_aux = FederatedDataset(
                dataset="uoft-cs/cifar100",
                partitioners={"train": partitioner_aux},
            )
        centralized_testset = fds_aux.load_split("test")
        pytorch_transforms_aux = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        def apply_transforms_aux(batch):
            batch["img"] = [pytorch_transforms_aux(img) for img in batch["img"]]
            return batch

        testset_aux = centralized_testset.with_transform(apply_transforms_aux)
        # Selecting test split w 500 samples for aux data
        testset_aux = torch.utils.data.Subset(testset_aux, [i for i in range(500)])
        testloader_aux = DataLoader(testset_aux, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Unsupported dataset for functional regularization: {wm_config.dataset_cl}")

    return testloader_aux


def label_reference(net):
    # Get the full model soup - pre-load all client states
    client_states = _load_all_client_states(wm_config.n_users, wm_config.folder)
    for cid_i in range(wm_config.n_users):
        client_model_params = [val for key_state, val in client_states[cid_i].items()]
        if cid_i == 0:
            average_model = [val/wm_config.n_users for val in client_model_params]
        else:
            average_model = [average_val + (val/wm_config.n_users) for average_val,val in zip(average_model,client_model_params)]
    set_weights(net,average_model)
    net.eval()
    # Load aux dataset
    auxloader = load_aux_data()

    all_predictions = []
    for data in auxloader:
        try:
            images = data["img"]
        except:
            images = data["image"]
        predictions = net(images.to(net.fc.weight.device))
        all_predictions.extend (predictions.detach().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    np.save(wm_config.folder + "aux_predictions.npy", all_predictions)
    
    return


## Evaluating evolution ##

def evaluate_config(server_round):
    # Run every time flwr launches evaluation
    # Adjust round number if we're resuming from checkpoint
    adjusted_round = server_round + wm_config.loading_round
    
    # Store models and triggers to disk every 250 rounds for analysis/checkpointing
    if adjusted_round % 250 == 0:
        shutil.copyfile(wm_config.folder + "triggers.npy", wm_config.folder + "trigger_round_"+ str(adjusted_round) +".npy")
        # Also checkpoint client states from Ray to disk
        for i_cid in range(wm_config.n_users):
            try:
                state_dict = _load_client_state(i_cid, wm_config.folder)
                # Save snapshot
                client_status = {"parameters": state_dict}
                with open(wm_config.folder + str(adjusted_round) + "_client_status_"+str(i_cid)+".pkl", 'wb') as f:
                    pickle.dump(client_status, f)
            except:
                # Fallback if state not in Ray yet
                try:
                    shutil.copyfile(wm_config.folder + "client_status_"+str(i_cid)+".pkl", 
                                  wm_config.folder + str(adjusted_round) + "_client_status_"+str(i_cid)+".pkl")
                except:
                    pass
    return {"current_round": adjusted_round, "server_round": server_round}


def check_metrics(net, i_cid, valloader, device):
    """Check metrics for the training evolution."""
    import time

    i_cid = int(i_cid)
    net.eval()

    ### Main task accuracy ###
    with torch.no_grad():
        loss, accuracy = test(net, valloader, device)

    ### Trigger accuracy ###
    if wm_config.trigger_type == "unique":
        triggers_np = np.load(wm_config.folder + "triggers.npy")[
            i_cid * wm_config.m : (i_cid + 1) * wm_config.m, :, :, :
        ]
        triggers = _normalize_triggers(triggers_np, device)
    else:
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy"), device)

    labels = wm_config.clients_tardos_q_tensor[i_cid, :].to(device)

    with torch.no_grad():
        outputs = net(triggers)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()

    t_accuracy = correct / wm_config.m

    ### MAV / collusion metrics ###
    # Wrap around so the last client colludes with client 0.
    col_cid = (i_cid + 1) % wm_config.n_users

    client_col_params = None
    last_err = None

    for _attempt in range(100):
        try:
            with open(wm_config.folder + "client_status_" + str(col_cid) + ".pkl", "rb") as f:
                client_col_status = pickle.load(f)
                client_col_params = client_col_status["parameters"]
            break
        except Exception as e:
            last_err = e
            time.sleep(0.1)

    if client_col_params is None:
        raise RuntimeError(
            f"Could not load colluder client_status_{col_cid}.pkl for cid={i_cid}"
        ) from last_err

    original_weights = get_weights(net)

    collusion_weights = [
        0.5 * (aux_cid.numpy() + curr_cid)
        for (_, aux_cid), curr_cid in zip(client_col_params.items(), original_weights)
    ]

    try:
        set_weights(net, collusion_weights)

        if wm_config.trigger_type == "unique":
            with torch.no_grad():
                outputs = net(triggers)
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                t_accuracy_c2 = correct / wm_config.m

            with open(wm_config.folder + "metrics_" + str(i_cid) + ".csv", "a") as file:
                file.write(f"{loss},{accuracy},{t_accuracy},{t_accuracy_c2}\n")

        else:
            with torch.no_grad():
                outputs = net(triggers)
                y = torch.argmax(outputs, dim=1).cpu().numpy()

                colluder_vectors = wm_config.clients_tardos_q[[i_cid, col_cid], :]
                mav = np.average([
                    y[i] not in colluder_vectors[:, i]
                    for i in range(wm_config.m)
                ])

                tp, t_s = tardos_accusation(
                    y,
                    vectors=wm_config.clients_tardos_q,
                    p_secret=wm_config.p_secret,
                    tau=wm_config.tau,
                )

                fn = tp == -1
                fp = (tp != -1) and (tp not in [i_cid, col_cid])

            with open(wm_config.folder + "metrics_" + str(i_cid) + ".csv", "a") as file:
                file.write(f"{loss},{accuracy},{t_accuracy},{mav},{fn},{fp}\n")

    finally:
        set_weights(net, original_weights)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return loss, len(valloader.dataset), accuracy


## Watermarking specific functions ##

def _save_client_state(cid, state_dict, folder):
    """Save a single client state to disk (for checkpointing)"""
    client_status = {"parameters": state_dict}
    with open(folder + "client_status_"+str(cid)+".pkl", 'wb') as f:
        pickle.dump(client_status, f)

def _load_client_state(cid, folder):
    """Load a single client state (from Ray if available, else disk)"""
    global _ray_client_states
    
    # Try Ray first (simulation mode)
    if wm_config.USE_RAY_STORE and cid in _ray_client_states:
        try:
            import ray
            return ray.get(_ray_client_states[cid])
        except:
            pass
    
    # Fallback to disk
    with open(folder + "client_status_"+str(cid)+".pkl", 'rb') as f:
        client_status = pickle.load(f)
        return client_status["parameters"]

def _store_client_state(cid, state_dict):
    """Store client state in Ray (simulation mode) or do nothing (disk mode)"""
    global _ray_client_states
    
    if wm_config.USE_RAY_STORE:
        try:
            import ray
            _ray_client_states[cid] = ray.put(state_dict)
        except:
            pass

def _load_all_client_states(n_users, folder):
    """Load all client states efficiently"""
    client_states = {}
    for i_cid in range(n_users):
        client_states[i_cid] = _load_client_state(i_cid, folder)
    return client_states

def _normalize_triggers(triggers_np, device):
    """Normalize numpy triggers to tensor"""
    triggers = torch.stack([TRIGGER_TRANSFORMS(Image.fromarray(trigger.astype(np.uint8))) for trigger in triggers_np]).float()
    return triggers.to(device)

def _compute_collusion_loss(net, triggers, labels_cid, labels, indexes, client_states, k_cols, criterion, p_secret, lambda_reg, device):
    """Compute collusion loss for a single client with memory-efficient forward passes"""
    original_weights = {name: param.clone() for name, param in net.named_parameters()}
    col_loss = 0.0
    
    for _ in range(k_cols):
        # Prepare collusion - reset to half original weights
        for name, param in net.named_parameters():
            param.data.copy_(original_weights[name].data / 2)
        
        # Select random colluder from pre-loaded states
        col_id = np.random.choice(indexes)
        labels_aux_id = labels[col_id, :]
        client_col_params_dict = client_states[col_id]
        client_col_params = [val for key_state, val in client_col_params_dict.items() if key_state in [key for key, val in net.named_parameters()]]
        
        # Add colluder weights
        for param, other_param in zip(net.parameters(), client_col_params):
            other_param = other_param.to(param.data.device)
            param.data += other_param / 2
        
        # Compute outputs once and compute both losses from single forward pass
        outputs = net(triggers)
        loss_cid = criterion(outputs, labels_cid)
        loss_aux = criterion(outputs, labels_aux_id)
        col_loss += lambda_reg * torch.mean(torch.where(loss_cid < loss_aux, loss_cid, loss_aux))
    
    return col_loss

def update_triggers(metrics):
    # Prepare model copies for reference (rename aux -> main)
    for i_cid in range(wm_config.n_users):
        try:
            os.rename(wm_config.folder + "aux_client_status_"+str(i_cid)+".pkl", wm_config.folder + "client_status_"+str(i_cid)+".pkl")
        except:
            pass

    t_rounds = wm_config.t_rounds
    k_cols = wm_config.k_cols
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if t_rounds > 0:
        """Aggregate triggers from all clients"""    
        
        # Load all client states (from Ray if available, else disk)
        print("Loading all client states...")
        client_states = _load_all_client_states(wm_config.n_users, wm_config.folder)
        
        # Load and normalize triggers
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy"), device)
        triggers_0 = _normalize_triggers(np.load(wm_config.folder + "trigger_round_0.npy"), device)
        
        # Load network
        if wm_config.model == "VGG16":
            net = models.VGG16()
        elif wm_config.model == "ResNet183x3":
            net = models.ResNet18()
        net.eval()
        net.to(device)
        
        # Prepare training
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        labels = wm_config.clients_tardos_q_tensor.to(device)
        p_secret = wm_config.p_secret_tensor.to(device)
        
        # Compute best triggers
        all_losses = []
        all_trigs = []
        all_trigs.append(triggers.cpu())

        for round in range(t_rounds):      
            total_loss = 0.0
            triggers.requires_grad = True
            
            for i_cid in range(wm_config.n_users):
                # Set original weights from pre-loaded state
                net.load_state_dict(client_states[i_cid])
                labels_cid = labels[int(i_cid), :]
                loss_base = torch.mean(criterion(net(triggers), labels_cid))
                
                # Add to loss
                total_loss += loss_base.item()
                loss_base.backward()
                
                # Compute collusion loss
                indexes = list(range(wm_config.n_users))
                indexes.pop(int(i_cid))
                
                loss_col = _compute_collusion_loss(
                    net, triggers, labels_cid, labels, indexes, client_states,
                    k_cols, criterion, p_secret, wm_config.lambda_reg, device
                )
                total_loss += float(loss_col)
                if float(loss_col) != 0.0:
                    loss_col.backward()

            # Also promote uniform output for the average of all models
            for cid_i in range(wm_config.n_users):
                client_model_params = [val for key_state, val in client_states[cid_i].items()]
                if cid_i == 0:
                    average_model = [val/wm_config.n_users for val in client_model_params]
                else:
                    average_model = [average_val + (val/wm_config.n_users) for average_val,val in zip(average_model,client_model_params)]
            set_weights(net,average_model)

            # Update triggers and fix constraints
            triggers = triggers - wm_config.t_step * triggers.grad.sign()
            triggers = torch.where(triggers > triggers_0 + wm_config.t_alpha, triggers_0 + wm_config.t_alpha, triggers)
            triggers = torch.where(triggers < triggers_0 - wm_config.t_alpha, triggers_0 - wm_config.t_alpha, triggers)
            triggers = torch.clamp(triggers, min=-1, max=1)
            triggers = triggers.detach()

            # Log triggers
            all_trigs.append(triggers.cpu())
            all_losses.append(total_loss)
            
            if round == t_rounds - 1:
                # Compute loss for last step
                total_loss = 0.0
                with torch.no_grad():
                    for i_cid in range(wm_config.n_users):
                        net.load_state_dict(client_states[i_cid])
                        labels_cid = labels[int(i_cid), :]
                        loss_base = torch.mean(criterion(net(triggers), labels_cid))
                        total_loss += loss_base.item()
                        
                        # Compute collusion loss
                        indexes = list(range(wm_config.n_users))
                        indexes.pop(int(i_cid))
                        original_weights = {name: param.clone() for name, param in net.named_parameters()}

                        for _ in range(k_cols):
                            for name, param in net.named_parameters():
                                param.data.copy_(original_weights[name].data / 2)
                            
                            col_id = np.random.choice(indexes)
                            labels_aux_id = labels[col_id, :]
                            client_col_params_dict = client_states[col_id]
                            client_col_params = [val for key_state, val in client_col_params_dict.items() if key_state in [key for key, val in net.named_parameters()]]
                            
                            for param, other_param in zip(net.parameters(), client_col_params):
                                other_param = other_param.to(param.data.device)
                                param.data += other_param / 2
                            
                            # Compute loss and select label with lowest loss
                            # Compute outputs once and compute both losses from single forward pass
                            outputs = net(triggers)
                            loss_cid = criterion(outputs, labels_cid)
                            loss_aux = criterion(outputs, labels_aux_id)
                            loss_col = wm_config.lambda_reg * torch.mean(torch.where(loss_cid < loss_aux, loss_cid, loss_aux))
                            total_loss += loss_col.item()
                
                all_losses.append(total_loss)

        # Save best triggers and fix the range
        # If there are not multiple trigger versions, no need to update
        if len(all_trigs) > 1:
            triggers = all_trigs[np.argmin(all_losses)]
            triggers = triggers * 0.5 + 0.5
            triggers = [ToPILImage()(trigger) for trigger in triggers]
            triggers = np.array(triggers).astype(np.uint8)
            np.save(wm_config.folder + "triggers.npy", triggers.astype(np.uint8))
    
    
    # Prepare label reference for next round
    if wm_config.lambda_cl > 0:
        # Load network
        if wm_config.model == "VGG16":
            net = models.VGG16()
        elif wm_config.model == "ResNet183x3":
            net = models.ResNet18()
        net.eval()
        net.to(device)
        label_reference(net)
        net.train()

    return {"triggers": 1}


def wm_client(net,i_cid,client_state,device,k_cols=wm_config.k_cols,epochs=1):
    """Watermarks model copy"""
    # Load model weights directly from ParametersRecord (more efficient)
    load_state_from_record(net, client_state)
    
    optim = torch.optim.SGD(net.parameters(), lr=wm_config.tlr, momentum=0.9, weight_decay=0.0001)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_CL = torch.nn.KLDivLoss(reduction='batchmean')
    
    net.train()
    net.to(device)
    # Freeze BN layers
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    # Load and normalize triggers once (not per epoch)
    if wm_config.trigger_type == "unique":
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy")[i_cid*wm_config.m:(i_cid+1)*wm_config.m,:,:,:], device)
    else:
        triggers = _normalize_triggers(np.load(wm_config.folder + "triggers.npy"), device)
    labels = wm_config.clients_tardos_q_tensor.to(device)
    p_secret = wm_config.p_secret_tensor.to(device)
    
    for _ in range(epochs):
        labels_cid = labels[int(i_cid),:]
        optim.zero_grad()
        loss_base = criterion(net(triggers), labels_cid)
        loss_col = 0
        indexes = list(range(wm_config.n_users))
        indexes.pop(int(i_cid))
        original_weights = {name: param.clone() for name, param in net.named_parameters()}

        for _ in range(k_cols):
            # Prepare collusion
            for name, param in net.named_parameters():
                param.data.copy_(original_weights[name].data / 2)
            
            # Select random colluder and load only its state (on-demand)
            col_id = np.random.choice(indexes)
            labels_aux_id = labels[col_id,:]
            client_col_params_dict = _load_client_state(col_id, wm_config.folder)
            client_col_params = [val for key_state, val in client_col_params_dict.items() if key_state in [key for key, val in net.named_parameters()]]

            for param, other_param in zip(net.parameters(), client_col_params):
                other_param = other_param.to(param.data.device)
                param.data += other_param / 2

            # Add to loss with lowest loss label selection
            # Compute outputs once and compute both losses from single forward pass
            outputs = net(triggers)
            loss_cid = criterion(outputs, labels_cid)
            loss_aux = criterion(outputs, labels_aux_id)
            loss_col += torch.where(loss_cid < loss_aux, loss_cid, loss_aux)

        # Restore original weights
        for name, param in net.named_parameters():
            param.data.copy_(original_weights[name].data)
        
        loss = torch.mean(loss_base + wm_config.lambda_reg * loss_col)
        loss.backward()

        # CL regularization
        if wm_config.lambda_cl > 0:
            try:
                cl_loss = 0
                # Try aux in case there are no aux outputs yet
                aux_labels = np.load(wm_config.folder + "aux_predictions.npy")
                # Load aux dataset
                auxloader = load_aux_data()
                
                for batch_i, data in enumerate(auxloader):
                    try:
                        images_aux = data["img"]
                    except:
                        images_aux = data["image"]
                    aux_labels_batch = aux_labels[batch_i*wm_config.mbs:(batch_i+1)*wm_config.mbs,:]
                    cl_loss += criterion_CL(F.log_softmax(net(images_aux.to(device)), dim=1), F.softmax(torch.from_numpy(aux_labels_batch).float().to(device), dim=1))

                # Calculate total loss and backpropagate
                loss = wm_config.lambda_cl * cl_loss

                loss.backward()
            except:
                pass

        optim.step()

    # Unfreeze batch norm layers
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            module.train()

    # Save watermarked weights and optimizer state
    p_record = ParametersRecord()
    for k, v in net.state_dict().items():
        # Convert to NumPy, then to Array. Add to record
        p_record[k] = array_from_numpy(v.detach().cpu().numpy())
    # Add to a context
    client_state.parameters_records["net_parameters"] = p_record

    return


## Leak and accusation functions ##

def load_collusion(user_idxs, model, folder = wm_config.folder, mode = "average", pruning = 0, fine = 0, fine_dataloader = None, ft_lr = wm_config.mlr, device = "cuda",before_cid="client_status_",after_cid=".pkl"):
    if mode == "average":
        for cid_i, cid in enumerate(user_idxs):
            # Load weights from each client
            with open(folder + before_cid +str(cid)+ after_cid, 'rb') as f:
                client_status = pickle.load(f) 
                client_model = client_status["parameters"]
                client_model = [val for key_state, val in client_model.items()]
            if cid_i == 0:
                average_model = [val/len(user_idxs) for val in client_model]
            else:
                average_model = [average_val + (val/len(user_idxs)) for average_val,val in zip(average_model,client_model)]
        set_weights(model,average_model)
    elif mode == "randomselect":    
        multip_model = [[] for _ in range(len(get_weights(model)))]
        layer_index = np.random.choice(user_idxs, len(get_weights(model)), replace=True)  
        for cid_i, cid in enumerate(user_idxs):
            # Load weights from each client
            with open(folder + before_cid +str(cid)+ after_cid, 'rb') as f:
                client_status = pickle.load(f) 
                client_model = client_status["parameters"]
                client_model = [val for key_state, val in client_model.items()]
            for layer, index in enumerate(layer_index):
                if index == cid:
                    multip_model[layer] = client_model[layer]
        set_weights(model,multip_model)
    if pruning != 0:
        # Pruning attack - move to CPU to avoid GPU memory issues
        model.cpu()
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning,
        )
        model.to(device)
    if fine != 0:
        # Fine-tuning
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=ft_lr)
        for _ in range(fine):
            if type(fine_dataloader) is list:
                for fine_dataloader_i in fine_dataloader:
                    for batch in fine_dataloader_i:
                        images = batch["img"]
                        try:
                            labels = batch["label"]
                        except:
                            labels = batch["fine_label"]
                        optimizer.zero_grad()
                        loss = criterion(model(images.to(device)), labels.to(device))
                        loss.backward()
                        optimizer.step()
            else:
                for batch in fine_dataloader:
                    images = batch["img"]
                    try:
                        labels = batch["label"]
                    except:
                        labels = batch["fine_label"]
                    optimizer.zero_grad()
                    loss = criterion(model(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
        model.eval()
    return model



def tardos_score(x,y,p):
  # Calculate the Tardos score for a secuence of outputs 
  # and a specific client vector
  t_score = 0
  for i_p in range(p.shape[0]):
    if y[i_p] == x[i_p]:
      t_score += np.sqrt((1-p[i_p,y[i_p]])/p[i_p,y[i_p]])
    else:
      t_score -= np.sqrt((p[i_p,y[i_p]])/(1-p[i_p,y[i_p]]))
  return t_score

def tardos_accusation(y, vectors, p_secret, tau, pfp=0.000001, trace_meta=None):
  import os
  import csv
  t_score = np.zeros(vectors.shape[0])

  trace_path = os.environ.get("BLACKCATT_TARDOS_TRACE", "")
  trace_meta = trace_meta or {}

  global _BLACKCATT_TARDOS_TRACE_CALL_IDX
  try:
    _BLACKCATT_TARDOS_TRACE_CALL_IDX += 1
  except NameError:
    _BLACKCATT_TARDOS_TRACE_CALL_IDX = 0

  call_idx = _BLACKCATT_TARDOS_TRACE_CALL_IDX
  run_tag = trace_meta.get("run", os.environ.get("BLACKCATT_RUN_TAG", "NA"))
  eval_client = trace_meta.get("eval_client", "NA")
  colluders = set(str(x) for x in trace_meta.get("colluders", []))

  first_tp = -1
  first_m_needed = vectors.shape[1]

  for mi in range(vectors.shape[1]):
    # Calculate threshold according to Skoric and Oosterwijk 2012
    a = 1 / (2 * np.log(pfp))
    b = 1 / (3 * np.sqrt(tau))
    c = mi + 1
    disc = b * b - 4 * a * c

    Z = np.nan
    if disc >= 0:
      x1 = (-b + np.sqrt(disc)) / (2 * a)
      x2 = (-b - np.sqrt(disc)) / (2 * a)
      if x1 > 0 and x2 > 0:
        Z = min([x1, x2])
      elif x1 > 0:
        Z = x1
      elif x2 > 0:
        Z = x2

    # Calculate the Tardos score for each client
    for client_index in range(vectors.shape[0]):
      inc = tardos_score(
        vectors[client_index, mi:mi+1],
        y[mi:mi+1],
        p_secret[mi:mi+1],
      )
      t_score[client_index] += float(np.asarray(inc).sum())

    # Original accusation condition
    accused_at_prefix = -1
    if np.isfinite(Z) and max(t_score) > Z:
      accused_at_prefix = int(np.argmax(t_score))

      # Preserve original first-crossing behavior
      if first_tp == -1:
        first_tp = accused_at_prefix
        first_m_needed = mi + 1

    # Extra logging, controlled by environment variable
    if trace_path:
      write_header = not os.path.exists(trace_path)

      with open(trace_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
          writer.writerow([
            "run",
            "call_idx",
            "eval_client",
            "prefix_len",
            "cid",
            "is_colluder",
            "score",
            "threshold",
            "margin",
            "accused_at_prefix",
            "first_tp_so_far",
            "pfp",
            "tau",
          ])

        for cid, score in enumerate(t_score):
          writer.writerow([
            run_tag,
            call_idx,
            eval_client,
            mi + 1,
            cid,
            int(str(cid) in colluders),
            float(score),
            float(Z),
            float(score - Z) if np.isfinite(Z) else np.nan,
            accused_at_prefix,
            first_tp,
            pfp,
            tau,
          ])

  return first_tp, first_m_needed