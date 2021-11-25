import torch
from torch import nn
from torchvision import models


class SkimageLoss:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, pred, true):
        bs = pred.size(0)
        res = 0
        for _pred, _true in zip(*map(self.routine, [pred, true])):
            res += self.metric(_true, _pred)

        return res / bs

    @staticmethod
    def routine(x):
        return x.permute(0, 2, 3, 1).detach().cpu().numpy()


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, multiscale_D=False):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.multiscale_D = multiscale_D
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            if isinstance(prediction[0], list):
                loss = 0
                for pred_i in prediction:
                    pred = pred_i[-1]
                    target_tensor = self.get_target_tensor(pred, target_is_real)
                    loss += self.loss(pred, target_tensor)
            else:
                target_tensor = self.get_target_tensor(prediction[-1], target_is_real)
                return self.loss(prediction[-1], target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction[-1].mean()
            else:
                loss = prediction[-1].mean()

        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self, n_layers_D, num_D):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.n_layers_D = n_layers_D
        self.num_D = num_D

    def forward(self, fake, real):
        loss = 0
        feat_weights = 4.0 / (self.n_layers_D + 1)
        d_weights = 1.0 / self.num_D
        for i in range(self.num_D):
            for j in range(len(fake[i])-1):
                loss += feat_weights * d_weights * self.criterion(fake[i][j], real[i][j].detach())

        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, pred, gt):
        n_channels = pred.shape[1]
        loss = 0
        for i in range(n_channels):
            _x, _y = self.inference_routine(pred[:, i, :, :]), self.inference_routine(gt[:, i, :, :])
            x_vgg, y_vgg = self.vgg(_x), self.vgg(_y)
            for w, x, y in zip(self.weights, x_vgg, y_vgg):
                loss += w * self.criterion(x, y.detach())

        return loss

    @staticmethod
    def inference_routine(x):
        _x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return (_x - 0.5) / 0.22


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
