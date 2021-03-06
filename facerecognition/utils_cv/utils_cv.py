#################################
# Nicolas LAUBE
# Sylvain MACE
# Melvin BICHO
# package utilitaire pour le projet
#####################################

import cv2
import matplotlib.pyplot as plt

Filepath = "../Data/tetris_blocks.png"


def load_and_display_image(filepath):
    """fonction permettant de charger et d'afficher une image (filename) """

    img = cv2.imread(filepath, 1)
    if img is not None:
        plot = plt.imshow(img)
        plt.show()
    else:
        raise ValueError("L'image " + filepath + " n'existe pas")


load_and_display_image(Filepath)


def process_image(filepath):
    img = cv2.imread(filepath, 0)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()


process_image(Filepath)


def process_image_2(filepath):

    img = cv2.imread(filepath, 1)
    edges = cv2.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img,cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()
