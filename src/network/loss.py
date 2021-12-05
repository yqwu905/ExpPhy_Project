import torch


class ContentLoss(torch.nn.Module):

    """Content Loss class"""

    def __init__(self, weight, target):
        """Init some settings

        :weight: weight matrix
        :target: Desire output

        """
        torch.nn.Module.__init__(self)

        self._weight = weight
        self._target = target.detach()*self._weight
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        """Forward method for Content Loss Layer

        :input: input matrix
        :returns: output matrix

        """
        self.loss = self.loss_fn(input*self._weight, self._target)
        self.output = input
        return self.output

    def backward(self):
        """Backward method for Content Loss Layer
        :returns: TODO

        """
        self.loss.backward(retain_graph = True)
        return self.loss


class GramMetrix(torch.nn.Module):

    """Gram Matrix Class"""

    def __init__(self):
        """Do nothing """
        torch.nn.Module.__init__(self)

    def forward(self, input):
        """Forward method for gram matrix

        :input: matrix
        :returns: TODO

        """
        a,b,c,d = input.size()
        feature = input.view(a*b, c*d)
        gram =torch.mm(feature, feature.t())
        return gram.div(a*b*c*d)
        

class StyleLoss(torch.nn.Module):

    """Style Loss Class"""

    def __init__(self, weight, target):
        """TODO: to be defined.

        :weight: weight matrix
        :target: Desire output
        """
        torch.nn.Module.__init__(self)
        self._weight = weight
        self._target = target.detach()*self._weight
        self._loss_fn = torch.nn.MSELoss()
        self._gram = GramMetrix()
    
    def forward(self, input):
        """Forward mehod for Style Loss

        :input: input matrix
        :returns: output matrix

        """
        self.output = input.clone()
        self.G= self._gram(input)
        self.G.mul_(self._weight)
        self.loss = self._loss_fn(self.G, self._target)
        return self.output

    def backward(self):
        """Backward method for Style loss
        :returns: Loss matrix

        """
        self.loss.backward(retain_graph = True)
        return self.loss
