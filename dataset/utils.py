from PIL import Image
from torchvision.transforms import Compose, ToTensor


def transform(img: Image.Image):
    trans = Compose([ToTensor()])
    return trans(img)
