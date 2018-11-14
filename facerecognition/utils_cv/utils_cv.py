#################################
# Nicolas LAUBE
# Sylvain MACE
# Melvin BICHO
# package utilitaire pour le projet
#####################################
import cv2
import matplotlib.pyplot as plt


def load_and_display_image(filepath):
    """fonction permettant de charger et d'afficher une image (filename) """

    img = cv2.imread(filepath, 1)
    if img is not None:
        plot = plt.imshow(img)
        plt.show()
    else:
        raise ValueError("L'image" + filepath + "n'existe pas")


load_and_display_image("C:/Users/melvi/Coding Weeks/G10_Raspberry_Pi_SNM/facerecognition/data/tetris_blocks.png")


