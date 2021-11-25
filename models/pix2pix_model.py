import torch
import torchio as tio
from torchio import DATA
from .base_model import BaseModel
from . import loss
from . import discriminator
from . import generator
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from functools import partial


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given
    paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters: parser (ArgumentParser) -- original option parser is_train (bool) -- whether training phase or
        test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'SSIM', 'PSNR']
        self.visual_names = ['fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = generator.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                       not opt.no_transp_conv, opt.n_layers_G, opt.act, not opt.no_preprocess_pointwise)

        if self.isTrain:
            self.netD = discriminator.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.criterionGAN = loss.GANLoss(opt.gan_mode, multiscale_D=opt.netD == 'pix2pixHD_multiscale').to(
                self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.criterionSSIM = loss.SkimageLoss(partial(ssim, multichannel=True))
            self.criterionPSNR = loss.SkimageLoss(psnr)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None
        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_D = None
        self.loss_G_L1 = None
        self.loss_G_GAN = None
        self.loss_G = None
        self.loss_SSIM = None
        self.loss_PSNR = None

    def set_input(self, _input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            _input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = _input['A' if AtoB else 'B'].to(self.device)
        self.real_B = _input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = _input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)

    def inference(self, patch_size=256, patch_stride=150):
        from dpipe.predict import patches_grid
        from dpipe.torch import inference_step
        import numpy as np

        @patches_grid(patch_size, patch_stride, axis=(-1, -2), padding_values=np.min)
        def inference(image):
            return inference_step(image, architecture=self.netG)

        self.fake_B = torch.from_numpy(inference(self.real_A.cpu().numpy()))

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.calc_metrics()

    @torch.no_grad()
    def calc_metrics(self):
        self.loss_SSIM = self.criterionSSIM(self.fake_B, self.real_B)
        self.loss_PSNR = self.criterionPSNR(self.fake_B, self.real_B)
