#################################
# Nicolas LAUBE
# Sylvain MACE
# Melvin BICHO
# package utilitaire pour le projet
#####################################
import unittest
from utils_cv.py import *

class Utils_test(unittest.TestCase):

    #def setUp(self):

    def test_tetris_blocks(self):
        self.assertTrue(load_and_display_image("C:/Users/Nicolas LAUBE/G10_Raspberry_Pi_SNM/facerecognition/Data/tetris_blocks.png") )


if __name__ == '__main__':
    unittest.main()
