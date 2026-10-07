from typing import List, Union
from models import Lenet5, Lenet300100, Lenet300, LenetLinear
import numpy as np
import torch
import torchvision
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Resize, Compose
import os
import sys
import shutil
from scipy.io import savemat
from pathlib import Path

UseRational = True
use_cuda = True
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
assert device.type in ["cpu", "cuda"]
print(device)

data_dir = "data"
Path(data_dir).mkdir(parents=True, exist_ok=True)
# activations_dir = f"{data_dir}/lenet-mnist/activations"



class Storage:
    def __init__(self):
        self.storage = {}
        self.hooks = []

    def __getitem__(self, key: str):
        return self.storage[key]

    def __setitem__(self, key: str, value: torch.Tensor):
        self.storage[key] = value

    def reset(self):
        for key, val in self.storage.items():
            self.storage[key] = []

    def _from_str(self, module, text: str):
        for name in text.split("."):
            module = getattr(module, name)
        return module

    def setup(self, module = None, iter_fn: callable = "named_children", layers: List[Union[torch.nn.Module, str]] = None):
        # either input the iter func or the layers as a list of modules or strings
        def _forward(module, input, output):
            self.storage[module.mod_name].append(output)

        # if you want to see what layers we are iterating over, breakpoint here
        if layers is not None:
            layers = [(layer, self._from_str(module, layer)) if isinstance(layer, str) else layer for layer in layers]
        else:
            iter_fn = getattr(module, iter_fn) if isinstance(iter_fn, str) else iter_fn
            layers = list(iter_fn())

        name_idx = 0
        for mod in layers:
            if isinstance(mod, torch.nn.Module):
                mod_name = f"{mod.__class__.__name__}_{name_idx}"
                name_idx += 1
            else:
                mod_name, mod = mod
            # name = mod_name + '_activation'
            mod.mod_name = mod_name
            self.storage[mod_name] = []
            hook = mod.register_forward_hook(_forward)
            self.hooks.append(hook)

    def __repr__(self):
        return f"Storage: {list(self.storage.keys())}"

    def saveable(self):
        out = {}
        for key, val in self.storage.items():
            if val == []:
                continue

            out[key] = torch.stack(val).cpu().detach().numpy()
        return out


if __name__ == '__main__':
    batch_size = 256
    N_CLASSES = 10
    n_epoch = 100
    if len(sys.argv) < 2:
        print("specify model: lenetlinear, lenet300, lenet300100, or lenet5")
        sys.exit(1)
    else:
        model_name = sys.argv[1]
    
    
    # ********************************** set-up model ************************************** 
    if model_name == "lenetlinear":
        model = LenetLinear(UseRational)
        model_path = None
    elif model_name == "lenet300":
        model = Lenet300(UseRational)
        model_path = None
    elif model_name == "lenet300100":
        model = Lenet300100(UseRational).to(device)
        model_path = "./data/Lenet300100/model/model_state_99.pt"
        model.load_state_dict(torch.load(model_path))
    elif model_name == "lenet5":
        model = Lenet5(UseRational).to(device)
        model_path = "./data/Lenet5/model/model_state_99.pt"
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Model: {model_name} not supported")
        sys.exit(1)
    model.eval()
    writer = SummaryWriter(f'runs/{model_name}_expr_1_alt_datasets')
    layers = list(model.named_children())
    cost = CrossEntropyLoss()

    # ******************************** setup Storage class for collecting layer activations ***********
    storage = Storage()
    # can do it like this:
    # storage.setup(layers=[model.layers.layer_0.linear, model.layers.layer_0.rat, model.layers.layer_1.linear, model.layers.layer_1.rat, model.layers.layer_2.linear])
    # or like  !!!! more general approach, might make deciphering saved activation outputs and inputs challenging !!!!
    storage.setup(model, iter_fn=model.named_modules)
    
    # ******************************** directory setup
    run_data_dir = f"{data_dir}/{model.__class__.__name__}"
    activations_dir = f"{run_data_dir}/activations"
    advgan_activations_dir = f"{activations_dir}/advgan"
    qmnist_activations_dir = f"{activations_dir}/qmnist"
    Path(activations_dir).mkdir(parents=True, exist_ok=True)
    Path(advgan_activations_dir).mkdir(parents=True, exist_ok=True)
    Path(qmnist_activations_dir).mkdir(parents=True, exist_ok=True)

    # ******************************* set-up for advgan dataset
    if model_name == "lenet300100":
        advgan_data_path = "/home/matthewmerris/repos/advGAN_pytorch/dataset/adv_mnist_test_lenet300100.pt"
    elif model_name == "lenet5":
        advgan_data_path = "/home/matthewmerris/repos/advGAN_pytorch/dataset/adv_mnist_test_lenet5.pt"
    else:
        print("advGAN dataset currently unavailable for specified model")
        sys.exit(1)
    
    advgan_dataset = torch.load(advgan_data_path,weights_only=False)
    advgan_loader = DataLoader(advgan_dataset, batch_size=batch_size, drop_last=True)
        

    # ******************************** save advGAN targets
    advgan_targets = []
    for dummy, batch in enumerate(advgan_loader):
        advgan_targets.append(batch[1])
    advgan_targets = torch.stack(advgan_targets, dim=0)
    advgan_targets = advgan_targets.numpy()
    np.save(f"{advgan_activations_dir}/targets.npy", advgan_targets)


    correct = 0
    seen = 0
    storage.reset()
    for idx, (test_x, test_label) in enumerate(advgan_loader):
        test_x, test_label = test_x.to(device), test_label.to(device)
        predict_y = model(test_x.float().to(device))

        predict_ys = predict_y.argmax(dim=-1)
        correct += (predict_ys == test_label).sum().item()
        seen += len(test_label)
        writer.add_scalar('Accuracy/test', correct / seen, idx)

        loss = cost(predict_y, test_label.long())
        writer.add_scalar('Loss/test', loss.sum().item(), idx)
    print('advGAN accuracy: {:.6f}'.format(correct / seen))
            
    save_data = storage.saveable()
    for key, item in save_data.items():
        np.save(f"{advgan_activations_dir}/{key}.npy", item)

    # ******************************** set up for qmnist dataset
    # Need to adjust for Lenet5 architecture
    if model_name == "lenet5":
        qmnist_dataset = torchvision.datasets.QMNIST(
            root=f"{data_dir}",
            what="test50k",
            download=True,
            transform=Compose([
                Resize((32,32)),
                ToTensor()])
        )
    elif model_name == "lenet300100":
        qmnist_dataset = torchvision.datasets.QMNIST(
            root=f"{data_dir}",
            what="test50k",
            download=True,
            transform=ToTensor()
        )    
    qmnist_loader = DataLoader(qmnist_dataset, batch_size=batch_size, drop_last=True)
    
    # ******************************** save qmnist targets
    '''
    qmnist_targets = []
    for dummy, batch in enumerate(qmnist_loader):
        qmnist_targets.append(batch[1])
    advgan_targets = torch.stack(qmnist_targets, dim=0)
    advgan_targets = qmnist_targets.numpy()
    np.save(f"{qmnist_activations_dir}/targets.npy", qmnist_targets)
    '''
    qmnist_targets = qmnist_dataset.targets.numpy
    np.save(f"{qmnist_activations_dir}/targets.npy", qmnist_targets)
    
    correct = 0
    seen = 0
    storage.reset()
    for idx, (test_x, test_label) in enumerate(qmnist_loader):
        test_x, test_label = test_x.to(device), test_label.to(device)
        predict_y = model(test_x.float().to(device))

        predict_ys = predict_y.argmax(dim=-1)
        correct += (predict_ys == test_label).sum().item()
        seen += len(test_label)
        writer.add_scalar('Accuracy/test', correct / seen, idx)

        loss = cost(predict_y, test_label.long())
        writer.add_scalar('Loss/test', loss.sum().item(), idx)
    print('QMNIST accuracy: {:.6f}'.format(correct / seen))
            
    save_data = storage.saveable()
    for key, item in save_data.items():
        np.save(f"{qmnist_activations_dir}/{key}.npy", item)
