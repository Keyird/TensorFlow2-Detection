from PIL import Image
from yolo import YOLO
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    yolo_tiny = YOLO()
    img = "img/dog.jpg"
    image = Image.open(img)
    r_image = yolo_tiny.detect_image(image)
    r_image.save("dog_predict.jpg")
    r_image.show()
