import torch
import torchvision
from torchvision import models
import copy
from network.loss import StyleLoss, GramMatrix, ContentLoss
from utils.image import load_img, save_img, show_img
from torch.autograd import Variable
from network.network import Network


def main(content_idx: int, style_idx: int):
    content_img = Variable(load_img(f"img/src/{content_idx}.png")).cuda()
    style_img = Variable(load_img(f"img/style/{style_idx}.png")).cuda()
    n = Network(content_img, style_img, 1, 1000)
    n.migrate()


if __name__ == "__main__":
    main(1, 5)
