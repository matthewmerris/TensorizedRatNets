from typing import List, Union
from models import Lenet5, Lenet300100, Lenet300, LenetLinear
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
import os
import sys
import shutil
from scipy.io import savemat

UseRational = True

data_dir = "data"
# activations_dir = f"{data_dir}/lenet-mnist/activations"
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
assert device in ["cpu", "cuda"]


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
    # ********************************** Transform dataset if necessary  **********************************
    train_dataset = mnist.MNIST(
        root=f'{data_dir}',
        train=True,
        transform=ToTensor(),
        # transform=Compose(
        #     [Resize(20), ToTensor()]
        # ),
        download=True
    )
    test_dataset = mnist.MNIST(
        root=f'{data_dir}',
        train=False,
        transform=ToTensor(),
        # transform=Compose(
        #     [Resize(20), ToTensor()]
        # ),
        download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    # ********************************** specify model to train **********************************
    model = Lenet300100(UseRational)  # Model(UseRational)
    # model = LenetLinear(UseRational)
    # model = Lenet300(UseRational)
    
    model.to(device)
    layers = list(model.named_children())
    # breakpoint()

    run_data_dir = f"{data_dir}/{model.__class__.__name__}"
    activations_dir = f"{run_data_dir}/activations"
    model_save_dir = f"{run_data_dir}/model"

    sgd = SGD(model.parameters(), lr=1e-1)
    cost = CrossEntropyLoss()
    n_epoch = 100

    # save test targets
    targets = []
    for dummy, batch in enumerate(test_loader):
        targets.append(batch[1])
    targets = torch.cat(targets, 0)
    targets = targets.numpy()
    # savemat(f"{activations_dir}/test/targets.mat", {"array":targets}, do_compression=False)
    np.save(f"{activations_dir}/test/targets.npy", targets)

    storage = Storage()
    # can do it like this:
    # storage.setup(layers=[model.layers.layer_0.linear, model.layers.layer_0.rat, model.layers.layer_1.linear, model.layers.layer_1.rat, model.layers.layer_2.linear])
    # or like  !!!! more general approach, might make deciphering saved activation outputs and inputs challenging !!!!
    storage.setup(model, iter_fn=model.named_modules)

    for epoch in range(n_epoch):
        model.train()
        storage.reset()
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            # breakpoint()
            train_x, train_label = train_x.to(device), train_label.to(device)
            # breakpoint()
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        epoch_save_dir_train = f"{activations_dir}/train/{epoch}"
        epoch_save_dir_test = f"{activations_dir}/test/{epoch}"

        # delete the old data, if its there
        if os.path.exists(epoch_save_dir_test):
            shutil.rmtree(epoch_save_dir_test)

        if os.path.exists(epoch_save_dir_train):
            shutil.rmtree(epoch_save_dir_train)

        # breakpoint()
        os.makedirs(epoch_save_dir_train, exist_ok=True)
        os.makedirs(epoch_save_dir_test, exist_ok=True)

        if epoch == n_epoch - 1:
            save_data = storage.saveable()
            for key, item in save_data.items():
                np.save(f"{epoch_save_dir_train}/{key}.npy", item)

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

        if epoch == n_epoch - 1:
            save_data = storage.saveable()
            for key, item in save_data.items():
                np.save(f"{epoch_save_dir_test}/{key}.npy", item)

        print('accuracy: {:.6f}'.format(correct / seen))

        os.makedirs(model_save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_save_dir}/model_state_{epoch}.pt")


