import argparse
import numpy as np
import matlab.engine
import matlab

from scipy.io import savemat, loadmat

## parse some arguments
parser = argparse.ArgumentParser(description="Lownerize activations matrices and decompose (Block-term)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--activations_in", help="Activation function inputs")
parser.add_argument("--activations_out", help="Activation function outputs")
parser.add_argument("--targets", help="Targets associated with specified activations")
args = parser.parse_args()
config = vars(args)

## open activation function inputs and outputs and targets
act_in = np.load(args.activations_in)
act_out = np.load(args.activations_out)
targets = np.load(args.targets)

## reshape into matrices
act_in = act_in.reshape((act_in.shape[0]*act_in.shape[1]), act_in.shape[2])
act_out = act_out.reshape((act_out.shape[0]*act_out.shape[1]), act_out.shape[2])

# breakpoint()
# split up activations by target and save as .mat files for matlab
num_targets = np.unique(targets).size
num_columns = act_in.shape[1]
data_path = "./tmp"
for i in range(num_targets):
    target_indices = np.where(targets == i)[0]
    tmp_act_in = np.zeros((target_indices.shape[0], num_columns))
    tmp_act_out = np.zeros((target_indices.shape[0], num_columns))
    row_idx = 0
    for trgt in target_indices:
        tmp_act_in[row_idx, :] = act_in[trgt, :]
        tmp_act_out[row_idx, :] = act_in[trgt, :]
        row_idx += 1
    print(f"Number of {i} targets: {row_idx}")
    in_path = f"{data_path}/in{args.activations_in.split('.')[1][-1]}_{i}s.mat"
    savemat(in_path, {"array": act_in}, do_compression=False)
    out_path = f"{data_path}/out{args.activations_out.split('.')[1][-1]}_{i}s.mat"
    savemat(out_path, {"array": act_out}, do_compression=False)

## fire up the matlab engine
# eng = matlab.engine.start_matlab("-nodesktop")
#
# ## Lownerize matrix rows
# loewnerized_act = []
# for i in range(act_in.shape[0]):
#     loewn = eng.loewnerize(act_out[i,:],act_in[i,:])
#     loewn = np.asarray(loewn)
#     loewnerized_act.append(loewn)
#
# ## stack into a mode-3 tensor and save to .mat format
# act_tensor = np.dstack(loewnerized_act)
# act_tensor_mini = act_tensor[:, :, 0:499]
# # savemat("act_tensor.mat", {"array": act_tensor}, do_compression=False)
#
# ## decompose (all in matlab, using tensorlab??)
# num_block_terms = 10
#
# block_rank = 3  #  we know the rational activation function is degree 3, ie. Loewner matrix is rank 3
# L = np.ones(num_block_terms, order='F') * block_rank
# breakpoint()
# Uhat = eng.ll1(np.ascontiguousarray(act_tensor_mini), L)
#
# breakpoint()