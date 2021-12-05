from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


transform = transforms.Compose([transforms.Scale([224,224]), transforms.ToTensor()])


def load_img(path:str):
    """Load image from given path

    :path:str, path of image to be loaded
    :returns: PIL image object

    """
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img


ToPIL = transforms.ToPILImage()


def save_img(img, path:str):
    """Save image

    :img: matrix of image in gpu
    :path:str, path of image to be saved.
    :returns: None

    """
    img = img.clone().cpu()
    img = img.view(3, 224, 224)
    img = ToPIL(img)
    img.save(path)


plt.ion()
plt.show()


def show_img(img, title=""):
    """Show image

    :img: matrix of image in gpu
    :title: str, title of plot, design to display iterations number.
    :returns: None

    """
    img = img.clone().cpu()
    img = img.view(3, 224, 224)
    img = ToPIL(img)
    plt.imshow(img)
    plt.title(title)
    # plt.draw()
    plt.show()
    plt.pause(0.1)
