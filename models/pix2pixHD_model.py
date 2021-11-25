import torch
from .base_model import BaseModel
from . import loss
from . import discriminator
from . import generator
from .pix2pix_model import Pix2PixModel
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from functools import partial


class Pix2PixHDModel(Pix2PixModel):
    """ This class implements the pix2pixHD model, for learning a mapping from input images to output images given
    paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pixHD paper: https://arxiv.org/pdf/1611.07004.pdf
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
            parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature-matching loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'Feat', 'VGG', 'SSIM', 'PSNR']
        self.visual_names = ['fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = generator.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                       not opt.no_transp_conv,
                                       opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                       opt.n_blocks_local)

        if self.isTrain:
            self.netD = discriminator.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'pix2pixHD_multiscale',
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                               not (opt.gan_mode == 'lsgan'), opt.num_D)

            self.criterionGAN = loss.GANLoss(opt.gan_mode, multiscale_D=opt.netD == 'pix2pixHD_multiscale').to(
                self.device)
            self.criterionVGG = loss.VGGLoss().to(self.device)
            self.criterionFeat = loss.FeatureMatchingLoss(opt.n_layers_D, opt.num_D)

            self.criterionSSIM = loss.SkimageLoss(partial(ssim, multichannel=True))
            self.criterionPSNR = loss.SkimageLoss(psnr)

            if opt.netG.startswith('pix2pixHD') and (opt.n_epochs_fix_global > 0):
                params_dict = dict(self.netG.named_parameters())
                netG_params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        netG_params += [value]
            else:
                netG_params = self.netG.parameters()

            self.optimizer_G = torch.optim.Adam(netG_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.epoch, pretrained_path)

        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None
        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_D = None
        self.loss_G_GAN = None
        self.loss_Feat = None
        self.loss_VGG = None
        self.loss_G = None
        self.loss_SSIM = None
        self.loss_PSNR = None

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        self.loss_G.backward()

    def backward(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_feat
        self.loss_Feat = self.criterionFeat(pred_fake, pred_real) * self.opt.lambda_feat

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_G = self.loss_G_GAN + self.loss_VGG + self.loss_Feat

    def optimize_parameters(self):
        self.forward()
        self.backward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.calc_metrics()

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
