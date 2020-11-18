import torch
import torch.nn as nn

from ..attack import Attack


class StepLL(Attack):
    r""" ILCM 迭代最小可能类法
    Least-Likely-Class Iterative Methods
    通过用识别概率最小的类别（目标类别）代替对抗扰动中的类别变量，再将原始图像减去该扰动，原始图像就变成了对抗样本，并能输出目标类别。
    iterative least-likely class attack in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 4/255)
        alpha (float): step size. (DEFALUT : 1/255)
        steps (int): number of steps. (DEFALUT : 0)
    
    .. note:: If steps set to 0, steps will be automatically decided following the paper.
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.StepLL(model, eps=4/255, alpha=1/255, steps=0)
        >>> adv_images = attack(images, labels)
        
    .. note:: Step-ll dosen't need any labels. However, for compatibility with other attacks, `labels` exists as input parameter in `forward` method. It is set to None by Default.
        
    """
    def __init__(self, model, eps=4/255, alpha=1/255, steps=0):
        super(StepLL, self).__init__("StepLL", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.to(self.device)

        outputs = self.model(images)
        _, labels = torch.min(outputs.data, dim=1) # 概率最小的类别作为标签
        labels = labels.detach_()

        loss = nn.CrossEntropyLoss()

        for i in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images - self.alpha*grad.sign() # 让预测结果朝着这个ll类走

            # 目测(猜测)一下，下面这里是为了把 一步eps分成很多小步的alpha
            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images + (a > adv_images).float()*a
            c = (b > images+self.eps).float()*(images+self.eps) + (images+self.eps >= b).float()*b
            images = torch.clamp(c, max=1).detach()

        adv_images = images

        return adv_images
