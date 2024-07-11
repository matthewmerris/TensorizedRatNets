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
            ll1_model.append(i * np.arange(12).reshape((3, 4)))

        testie = ll1_tools.unpack_ll1(ll1_model)

        self.assertEqual(ll1_model[0][0, 0], testie[0][0][0])
        self.assertEqual(ll1_model[0][1, 0], testie[0][1][0])
        self.assertEqual(ll1_model[1][1, 0], testie[0][1][1])
        self.assertEqual(ll1_model[1][0, 0], testie[1][0][0])
        self.assertEqual(ll1_model[1][1, 1], testie[1][1][1])

    def test_pack_ll1(self):
        ll1_model = list()
        for i in range(3):
            ll1_model.append(i * np.arange(12).reshape((3, 4)))

        testie = ll1_tools.unpack_ll1(ll1_model)
        # testie = ll1_tools.convert_ll1(testie, "matlab")
        testie = ll1_tools.pack_ll1(testie)

        self.assertEqual(ll1_model[0][0, 0], testie[0][0, 0])



if __name__ == '__main__':
    unittest.main()
