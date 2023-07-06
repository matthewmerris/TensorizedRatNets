import argparse
import numpy as np



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

## Lownerize matrix rows

## stack into a mode-3 tensor

## decompose (all in matlab, using tensorlab??)
