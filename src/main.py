import torch
import torchvision
from torchvision import models
import copy
from network.loss import StyleLoss, GramMetrix, ContentLoss
from utils.image import load_img, save_img, show_img
from torch.autograd import Variable

content_idx, style_idx = 2,6
# Load image
content_img = load_img(f"img/src/{content_idx}.png")
content_img = Variable(content_img).cuda()
style_img = load_img(f"img/style/{style_idx}.png")
style_img = Variable(style_img).cuda()

# Load vgg16
use_gpu = torch.cuda.is_available()
cnn = models.vgg16(pretrained=True).features
if use_gpu:
    cnn = cnn.cuda()
print(cnn)

# Modify vgg16

content_layer = ["Conv_5", "Conv_6"]
style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]
content_losses = []
style_losses = []
content_weight = 1
style_weight = 1000
new_model = torch.nn.Sequential()
model = copy.deepcopy(cnn)
gram = GramMetrix()
if use_gpu:
    new_model = new_model.cuda()
    gram = gram.cuda()
index = 1
for layer in list(model):
    if isinstance(layer, torch.nn.Conv2d):
        name = "Conv_" + str(index)
        new_model.add_module(name, layer)
        if name in content_layer:
            target = new_model(content_img).clone()
            content_loss = ContentLoss(content_weight, target)
            new_model.add_module(f"content_loss_{index}", content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = StyleLoss(style_weight, target)
            new_model.add_module(f"style_loss_{index}", style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, torch.nn.ReLU):
        name = f"ReLu_{index}"
        new_model.add_module(name, layer)
        index += 1

    if isinstance(layer, torch.nn.MaxPool2d):
        name = f"MaxPool_{index}"
        new_model.add_module(name, layer)

print(new_model)

# Prepare image and optimizer
input_img = content_img.clone()
parameter = torch.nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter])


n_epoch = 1000

run = [0]
while run[0] <= n_epoch:

    def closure():
        """Closure for optimizer
        :returns: TODO

        """
        optimizer.zero_grad()
        style_score = 0
        content_score = 0
        parameter.data.clamp_(0, 1)
        new_model(parameter)
        for sl in style_losses:
            style_score += sl.backward()

        for cl in content_losses:
            content_score += cl.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f"{run[0]} Style Loss : {style_score}, {content_score}")
            parameter.data.clamp_(0,1)
            show_img(parameter.data, f"Iter {run[0]}")
        return style_score + content_score
    
    optimizer.step(closure)

parameter.data.clamp_(0,1)
save_img(parameter.data, f"{content_idx}-{style_idx}.png")
