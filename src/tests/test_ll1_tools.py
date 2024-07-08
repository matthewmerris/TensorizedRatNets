import unittest
import sys
sys.path.append('../../src/')
import os
import ll1_tools
import matlab
import matlab.engine
import numpy as np


class TestLL1Tools(unittest.TestCase):
    def test_unpack_ll1(self):
        ll1_model = list()
        for i in range(3):
            ll1_model.append(i * np.arange(24).reshape((4,6)))

        testie = ll1_tools.unpack_ll1(ll1_model)

        self.assertEquals(ll1_model[0][0,0], testie[0][0][0])


if __name__ == '__main__':
    unittest.main()
    