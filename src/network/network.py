import torch
import torchvision
from torchvision import models
from network.loss import StyleLoss, GramMatrix, ContentLoss
from utils.image import save_img, show_img


class Network(object):

    """Network class for image style migration"""

    def __init__(
        self, content_img, style_img,
        content_weight: int, style_weight: int
    ):
        """Initialize

        :content_img: Content image
        :style_img: Style image
        :content_weight: Weight of content loss
        :style_weight: Weight of style loss

        """
        self._content_img = content_img
        self._style_img = style_img
        self._content_weight = content_weight
        self._style_weight = style_weight
        self._content_losses = []
        self._style_losses = []
        self._cnn = torch.nn.Sequential()
        vgg16 = models.vgg16(pretrained=True).features
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            vgg16 = vgg16.cuda()
            self._cnn = self._cnn.cuda()
        content_layer = ["Conv_5", "Conv_6"]
        style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]
        gram = GramMatrix()
        if use_gpu:
            gram = gram.cuda()
        idx = 1
        for layer in list(vgg16):
            if isinstance(layer, torch.nn.Conv2d):
                name = f"Conv_{idx}"
                self._cnn.add_module(name, layer)
                if name in content_layer:
                    target = self._cnn(self._content_img).clone()
                    c = ContentLoss(content_weight, target)
                    self._cnn.add_module(f"content_loss_{idx}", c)
                    self._content_losses.append(c)

                if name in style_layer:
                    target = self._cnn(style_img).clone()
                    target = gram(target)
                    s = StyleLoss(self._style_weight, target)
                    self._cnn.add_module(f"style_loss_{idx}", s)
                    self._style_losses.append(s)

            if isinstance(layer, torch.nn.ReLU):
                self._cnn.add_module(f"ReLU_{idx}", layer)
                idx += 1

            if isinstance(layer, torch.nn.MaxPool2d):
                self._cnn.add_module(f"MaxPool_{idx}", layer)
        print(self._cnn)

    def migrate(
        self, max_epoch: int = 1000, print_iter: int = 50,
        is_show_img: bool = True, is_save_internal_img: bool = True
    ):
        """Migrate image style
        :max_epoch: Max training epoch
        :print_iter: Print loss after how many epoch
        :is_show_img: Whether show internal migrate image while print loss
        :is_save_internal_img: Whether save internal migrate image while
        print loss
        :returns: None

        """
        input_img = self._content_img.clone()
        parameter = torch.nn.Parameter(input_img.data)
        optimizer = torch.optim.LBFGS([parameter])

        i = [0]
        while i[0] < max_epoch:
            def closure():
                """Closure fr optimizer
                :returns: total score

                """
                optimizer.zero_grad()
                style_score = 0
                content_score = 0
                parameter.data.clamp_(0, 1)
                self._cnn(parameter)
                for sl in self._style_losses:
                    style_score += sl.backward()

                for cl in self._content_losses:
                    content_score += cl.backward()

                i[0] += 1
                if i[0] % print_iter == 0:
                    print(
                        f"{i[0]} Style Loss: {style_score}, "
                        f"Content Loss: {content_score}")
                    if is_show_img:
                        show_img(parameter.data, f"Iter {i[0]}")
                    if is_save_internal_img:
                        save_img(parameter.data, f"img/output/{i[0]}.png")
                return style_score + content_score

            optimizer.step(closure)

        parameter.data.clamp_(0, 1)
        save_img(parameter.data, f"img/output/out.png")
