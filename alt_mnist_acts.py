from typing import List, Union
from models import Lenet5, Lenet300100, Lenet300, LenetLinear
import numpy as np
import torch
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
        model = Lenet300100(UseRational)
        model_path = "./data/Lenet300100/model/model_state_99.pt"
    elif model_name == "lenet5":
        model = Lenet5(UseRational)
        model_path = "./data/Lenet5/model/model_state_99.pt"
    else:
        print(f"Model: {model_name} not supported")
        sys.exit(1)
    writer = SummaryWriter(f'runs/{model_name}_expr_1')
        
    model.to(device)
    layers = list(model.named_children())
    
    # log model graph
#    dummy_input = train_loader.dataset[0]
#    print(dummy_input)
#    writer.add_graph(model, dummy_input[0])
    
    # ********************************* specify loss criterion (cost) & optimizer (SGD) 
    optimizer = SGD(model.parameters(), lr=1e-1)
    cost = CrossEntropyLoss()
    # breakpoint()

    # ******************************** directory setup
    run_data_dir = f"{data_dir}/{model.__class__.__name__}"
    activations_dir = f"{run_data_dir}/activations"
    Path(activations_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{activations_dir}/test").mkdir(parents=True, exist_ok=True)
    model_save_dir = f"{run_data_dir}/model"
    Path(activations_dir).mkdir(parents=True, exist_ok=True)


    # ******************************** save test targets
    targets = []
    for dummy, batch in enumerate(test_loader):
        targets.append(batch[1])
    targets = torch.cat(targets, 0)
    targets = targets.numpy()
    # savemat(f"{activations_dir}/test/targets.mat", {"array":targets}, do_compression=False)
    np.save(f"{activations_dir}/test/targets.npy", targets)

    # ******************************** setup Storage class for collecting layer activations ***********
    storage = Storage()
    # can do it like this:
    # storage.setup(layers=[model.layers.layer_0.linear, model.layers.layer_0.rat, model.layers.layer_1.linear, model.layers.layer_1.rat, model.layers.layer_2.linear])
    # or like  !!!! more general approach, might make deciphering saved activation outputs and inputs challenging !!!!
    storage.setup(model, iter_fn=model.named_modules)

    for epoch in range(n_epoch):
        # ********************* TRAIN ********************************
        if epoch % 10 == 0:
            print(f'Begin epoch: {epoch}')
        correct = 0
        seen = 0
        model.train()
        storage.reset()
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            # breakpoint()
            train_x, train_label = train_x.to(device), train_label.to(device)
            # breakpoint()
            optimizer.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            predict_ys = predict_y.argmax(dim=-1)
            correct += (predict_ys == train_label).sum().item()
            seen += len(train_label)             
            if idx % 100 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.sum().item(), epoch)

        writer.add_scalar('Accuracy/train', correct / seen, epoch)
        epoch_save_dir_train = f"{activations_dir}/train/{epoch}"
        epoch_save_dir_test = f"{activations_dir}/test/{epoch}"

        # delete the old data, if its there
        if os.path.exists(epoch_save_dir_test):
            shutil.rmtree(epoch_save_dir_test)

        if os.path.exists(epoch_save_dir_train):
            shutil.rmtree(epoch_save_dir_train)

        # breakpoint()
#        os.makedirs(epoch_save_dir_train, exist_ok=True)
#        os.makedirs(epoch_save_dir_test, exist_ok=True)

        if epoch == n_epoch - 1:
            os.makedirs(epoch_save_dir_train, exist_ok=True)
            os.makedirs(epoch_save_dir_test, exist_ok=True)
            save_data = storage.saveable()
            for key, item in save_data.items():
                np.save(f"{epoch_save_dir_train}/{key}.npy", item)

        # ********************* TEST ********************************
        correct = 0
        seen = 0

        model.eval()
        storage.reset()
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x, test_label = test_x.to(device), test_label.to(device)
            predict_y = model(test_x.float().to(device))

            predict_ys = predict_y.argmax(dim=-1)
            correct += (predict_ys == test_label).sum().item()
            seen += len(test_label)
            writer.add_scalar('Accuracy/test', correct / seen, epoch)

            loss = cost(predict_y, test_label.long())
            writer.add_scalar('Loss/test', loss.sum().item(), epoch)

        if epoch == n_epoch - 1:
            save_data = storage.saveable()
            for key, item in save_data.items():
                np.save(f"{epoch_save_dir_test}/{key}.npy", item)

        print('accuracy: {:.6f}'.format(correct / seen))

        os.makedirs(model_save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_save_dir}/model_state_{epoch}.pt")

    # ********************* close the tensorboard writer ****************************************
    writer.flush()
    writer.close()
