import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.utils import save_image

models_path = './models/'
Tensor = torch.cuda.FloatTensor

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class WGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                             lr=3e-4, betas=(0.5, 0.9))
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                             lr=1e-4, betas=(0.5, 0.9))



    def train(self, train_dataloader, epochs):
        lossD_array = np.empty(epochs)
        lossG_array = np.empty(epochs)
        lossP_array = np.empty(epochs)
        lossA_array = np.empty(epochs)
        lossG_all_array = np.empty(epochs)

        for epoch in range(1, epochs+1):
            
            for i, data in enumerate(train_dataloader, start=0):

                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                images = Variable(images.type(Tensor))
                
                
                w = self.netG(images)
                adv_images = (torch.tanh(w / 15) + 1) / 2

                self.optimizer_D.zero_grad()

                # Real images
                real_validity = self.netDisc(images)
                # Fake images
                fake_validity = self.netDisc(adv_images.detach())
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(self.netDisc, images.data, adv_images.data)
                # Adversarial loss
                loss_D_GAN = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
       
                loss_D_GAN.backward()
                self.optimizer_D.step()

            # optimize G
                if i % 5 == 0:

                    self.optimizer_G.zero_grad()
                    fake_validity = self.netDisc(adv_images)
                    loss_G_fake = -torch.mean(fake_validity)

                    # calculate perturbation norm
                    C = 0.1
                    perturbation = adv_images - images
                    loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
                    loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

                    # cal adv loss
                    logits_model = self.model(adv_images)
                    probs_model = F.softmax(logits_model, dim=1)
                    
                    #non-target
                    onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
                    #targeted
                    # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[torch.ones_like(labels)*0]

                    # C&W loss function
                    real = torch.sum(onehot_labels * probs_model, dim=1) #prob of real label  
                    other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1) #max prob of label != real  
                    zeros = torch.zeros_like(other)
                    
                    #non-target
                    loss_adv = torch.max(real - other, zeros)
                    loss_adv = torch.sum(loss_adv)                    
                    #targeted
                    # loss_adv = torch.max(other - real, zeros)
                    # loss_adv = torch.sum(loss_adv)
                    
                    #alpha=5 beta=17 gamma=1
                    adv_lambda = 1
                    pert_lambda = 17
                    pert_g = 5
                    loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb + pert_g * loss_G_fake
                    loss_G.backward()
                    self.optimizer_G.step()

            save_image(adv_images.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
            \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                (epoch, loss_D_GAN, loss_G_fake,
                loss_perturb, loss_adv))

            lossD_array[epoch - 1] = loss_D_GAN
            lossG_array[epoch - 1] = loss_G_fake
            lossP_array[epoch - 1] = loss_perturb
            lossA_array[epoch - 1] = loss_adv
            lossG_all_array[epoch - 1] = loss_G
        
        netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
        torch.save(self.netG.state_dict(), netG_file_name)
        
        plt.figure()
        plt.xlim(0,epoch)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(lossA_array)
        plt.plot(lossP_array)
        plt.plot(lossD_array)
        plt.plot(lossG_array)
        plt.legend(['loss_adv','loss_hinge', 'loss_D', 'loss_G_gan'])
        plt.savefig('loss.png')

        plt.figure()
        plt.xlim(0,epoch)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(lossD_array)
        plt.plot(lossG_all_array)
        plt.legend(['loss_D','loss_G'])
        plt.savefig('loss_DG.png')

        plt.close('all')
        
