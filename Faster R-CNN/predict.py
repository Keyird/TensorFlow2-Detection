from PIL import Image
from frcnn import FRCNN
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    frcnn = FRCNN()
    img = "img/dog.jpg"
    image = Image.open(img)
    r_image = frcnn.detect_image(image)
    r_image.show()
