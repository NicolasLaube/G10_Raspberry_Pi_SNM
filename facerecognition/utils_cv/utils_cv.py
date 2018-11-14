#################################
# Nicolas LAUBE
# Sylvain MACE
# Melvin BICHO
# package utilitaire pour le projet
#####################################
import cv2
import os


def load_and_display_image(filepath):
    """fonction permettant de charger et d'afficher une image (filename) """

    img = cv2.imread(filepath, 1)
    if img is not None:
        cv2.imshow('img', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError("L'image" + filepath + "n'existe pas")
print()


load_and_display_image(os.path.expanduser('~')+ "/G10_Raspberry_Pi_SNM/facerecognition/Data/tetris_blocks.png")


