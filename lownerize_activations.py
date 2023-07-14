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
args = parser.parse_args()
config = vars(args)

## open activation function inputs and outputs
act_in = np.load(args.activations_in)
act_out = np.load(args.activations_out)

## reshape into matrices
act_in = act_in.reshape((act_in.shape[0]*act_in.shape[1]), act_in.shape[2])
act_out = act_out.reshape((act_out.shape[0]*act_out.shape[1]), act_out.shape[2])

## convert save as .mat files

## fire up the matlab engine
eng = matlab.engine.start_matlab("-nodesktop")

## Lownerize matrix rows
loewnerized_act = []
for i in range(act_in.shape[0]):
    loewn = eng.loewnerize(act_out[i,:],act_in[i,:])
    loewn = np.asarray(loewn)
    loewnerized_act.append(loewn)

## stack into a mode-3 tensor and save to .mat format
act_tensor = np.dstack(loewnerized_act)
act_tensor_mini = act_tensor[:, :, 0:499]
# savemat("act_tensor.mat", {"array": act_tensor}, do_compression=False)

## decompose (all in matlab, using tensorlab??)
num_block_terms = 10

block_rank = 3  #  we know the rational activation function is degree 3, ie. Loewner matrix is rank 3
L = np.ones(num_block_terms, order='F') * block_rank
breakpoint()
Uhat = eng.ll1(np.ascontiguousarray(act_tensor_mini), L)

breakpoint()