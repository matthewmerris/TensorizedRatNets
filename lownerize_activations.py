import argparse
import numpy as np
import matlab.engine

## parse some arguments
parser = argparse.ArgumentParser(description="Lownerize activations matrices and decompose (Block-term)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("activations_in", help="Activation function inputs")
parser.add_argument("activations_out", help="Activation function outputs")
args = parser.parse_args()
config = vars(args)

## open activation function inputs and outputs
act_in = np.load(args["activations_in"])
act_out = np.load(args["activations_out"])

## reshape into matrices
act_in = act_in.reshape((act_in.shape[0]*act_in.shape[1]), act_in.shape[2])
act_out = act_out.reshape((act_out.shape[0]*act_out.shape[1]), act_out.shape[2])

## fire up the matlab engine
eng = matlab.engine.start_matlab("-nodesktop", "-nojvm")

## Lownerize matrix rows and stack into a order-3 tensor
loewnerized_act = np.empty([act_in.shape[1]/2,act_in.shape[1]/1,act_in.shape[0]])
for i in range(act_in.shape[0]):
    loewn = eng.loewnerize(act_out[i,:],act_in[i,:])
    loewn = np.asarray(loewn)





## stack into a mode-3 tensor

## decompose (all in matlab, using tensorlab??)
