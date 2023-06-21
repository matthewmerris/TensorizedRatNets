from scipy.io import savemat
import numpy as np
import glob
import os

npyFiles = glob.glob("*.npy")
for f in npyFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f)
    savemat(fm,{'d':d})
    print('generated ', fm, ' from ', f)
