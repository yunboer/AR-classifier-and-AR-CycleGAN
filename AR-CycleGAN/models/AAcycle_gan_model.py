import torch
import itertools
from util.image_pool import ImagePool
from util.util import SSIM_CAM, SSIM, BSC
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F


class AACycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_rsna', type=float, default=1.0, help='loss with rsna')
            parser.add_argument('--lambda_SC_A', type=float, default=1.0, help='loss with SC')
            parser.add_argument('--lambda_SC_B', type=float, default=1.0, help='loss with SC')
            parser.add_argument('--lambda_SC1', type=float, default=1.0, help='loss with SC')
            parser.add_argument('--lambda_SC2', type=float, default=1.0, help='loss with SC')
            parser.add_argument('--lambda_BSC_sts', type=float, default=1.0, help='loss with SC')
            parser.add_argument('--lambda_BSC_tst', type=float, default=1.0, help='loss with SC')
            parser.add_argument('--lambda_SC4_A', type=float, default=1.0, help='loss with SC')
            parser.add_argument('--lambda_SC4_B', type=float, default=1.0, help='loss with SC')
            

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B','SC1', 'SC2','BSC_sts','BSC_tst','SC4']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']

        
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc+1, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_R_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_R_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSCCAM = SSIM_CAM()
            self.criterionSC = SSIM()
            self.criterionBSC = BSC()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.real_A_R = input['A_R' if AtoB else 'B_R'].to(self.device)
        self.real_B_R = input['B_R' if AtoB else 'A_R'].to(self.device)
        
        self.cam_sts = input['cam_sts' if AtoB else 'cam_tst'].to(self.device)
        self.cam_tst = input['cam_tst' if AtoB else 'cam_sts'].to(self.device)
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print(self.real_A.shape)
        # print(self.cam_sts.shape)
        self.fake_B = self.netG_A(torch.concat([self.real_A,self.cam_sts.unsqueeze(1)],dim=1))  # G_A(A)
        self.rec_A = self.netG_B(torch.concat([self.fake_B,self.cam_sts.unsqueeze(1)],dim=1))   # G_B(G_A(A))
        self.fake_A = self.netG_B(torch.concat([self.real_B,self.cam_tst.unsqueeze(1)],dim=1))  # G_B(B)
        self.rec_B = self.netG_A(torch.concat([self.fake_A,self.cam_tst.unsqueeze(1)],dim=1))   # G_A(G_B(B))
        
        self.fake_B_R = self.netG_A(torch.concat([self.real_A_R,self.cam_sts.unsqueeze(1)],dim=1))  # G_A(A)
        self.rec_A_R = self.netG_B(torch.concat([self.fake_B_R,self.cam_sts.unsqueeze(1)],dim=1))   # G_B(G_A(A))
        self.fake_A_R = self.netG_B(torch.concat([self.real_B_R,self.cam_tst.unsqueeze(1)],dim=1))  # G_B(B)
        self.rec_B_R = self.netG_A(torch.concat([self.fake_A_R,self.cam_tst.unsqueeze(1)],dim=1))   # G_A(G_B(B))
        
        

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B_R = self.fake_B_R_pool.query(self.fake_B_R)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) +\
                        self.backward_D_basic(self.netD_A, self.real_B_R, fake_B_R) * self.opt.lambda_rsna
        # self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B) +self.backward_D_basic(self.netD_A, self.real_B_R, self.fake_B_R) * self.opt.lambda_rsna


    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        fake_A_R = self.fake_A_R_pool.query(self.fake_A_R)
        
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) +\
                        self.backward_D_basic(self.netD_B, self.real_A_R, fake_A_R) * self.opt.lambda_rsna
        
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A) + self.backward_D_basic(self.netD_B, self.real_A_R, self.fake_A_R) * self.opt.lambda_rsna

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_rsna = self.opt.lambda_rsna
        lambda_SC_A = self.opt.lambda_SC_A
        lambda_SC_B = self.opt.lambda_SC_B
        lambda_SC2 = self.opt.lambda_SC2
        lambda_SC1 = self.opt.lambda_SC1
        lambda_BSC_sts = self.opt.lambda_BSC_sts
        lambda_BSC_tst = self.opt.lambda_BSC_tst
        # lambda_SC4_A = self.opt.lambda_SC4_A
        # lambda_SC4_B = self.opt.lambda_SC4_B
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) + lambda_rsna * self.criterionGAN(self.netD_A(self.fake_B_R), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) + lambda_rsna * self.criterionGAN(self.netD_B(self.fake_A_R), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = (self.criterionCycle(self.rec_A, self.real_A) + lambda_rsna * self.criterionCycle(self.rec_A_R, self.real_A_R))* lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = (self.criterionCycle(self.rec_B, self.real_B) + lambda_rsna * self.criterionCycle(self.rec_B_R, self.real_B_R))* lambda_B

        # SSIM between real and fake with cam as a mask
        self.loss_SC1_A = F.l1_loss(self.criterionSCCAM(self.real_A,self.real_A_R,self.cam_sts),
                                    self.criterionSCCAM(self.fake_B,self.fake_B_R,self.cam_sts)) * lambda_SC_A
        self.loss_SC1_B = F.l1_loss(self.criterionSCCAM(self.real_B,self.real_B_R,self.cam_tst),
                                    self.criterionSCCAM(self.fake_A,self.fake_A_R,self.cam_tst)) * lambda_SC_B
        
        self.loss_SC1 = (self.loss_SC1_A + self.loss_SC1_B)/2 * lambda_SC1
        
        # SSIM between real and rec
        self.loss_SC2_A = self.criterionSC(self.real_A,self.rec_A)
        self.loss_SC2_A_R = self.criterionSC(self.real_A_R,self.rec_A_R)
        
        self.loss_SC2_B = self.criterionSC(self.real_B,self.rec_B)
        self.loss_SC2_B_R = self.criterionSC(self.real_B_R,self.rec_B_R)
        
        self.loss_SC2 = self.loss_SC2_A     +\
                        self.loss_SC2_A_R   +\
                        self.loss_SC2_B     +\
                        self.loss_SC2_B_R
        
        self.loss_SC2 = - self.loss_SC2 /4 * lambda_SC2
        
        # L1-loss between real and fake with cam masked
        self.loss_BSC_sts = (self.criterionBSC(self.real_A, self.fake_B,self.cam_sts) +\
                            self.criterionBSC(self.real_A_R, self.fake_B_R, self.cam_sts) * lambda_rsna) * lambda_BSC_sts
        self.loss_BSC_tst = (self.criterionBSC(self.real_B, self.fake_A,self.cam_tst) +\
                            self.criterionBSC(self.real_B_R, self.fake_A_R, self.cam_tst) * lambda_rsna) * lambda_BSC_tst
        
        self.loss_BSC = self.loss_BSC_sts + self.loss_BSC_tst
        
        # SSIM-loss between real and fake with cam masked
        # self.loss_SC4_A = (self.criterionSCCAM(self.real_A, self.fake_B,self.cam_sts) +\
        #                     self.criterionSCCAM(self.real_A_R, self.fake_B_R, self.cam_sts) * lambda_rsna) * lambda_SC4_A
        # self.loss_SC4_B = (self.criterionSCCAM(self.real_B, self.fake_A,self.cam_tst) +\
        #                     self.criterionSCCAM(self.real_B_R, self.fake_A_R, self.cam_tst) * lambda_rsna) * lambda_SC4_B
        
        # self.loss_SC4 = -(self.loss_SC4_A + self.loss_SC4_B)/4
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B +\
                    self.loss_cycle_A + self.loss_cycle_B +\
                    self.loss_idt_A + self.loss_idt_B +\
                    self.loss_SC1 +\
                    self.loss_SC2 +\
                    self.loss_BSC 
                    
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
