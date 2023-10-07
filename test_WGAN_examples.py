import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,TensorDataset
import models
from models import MNIST_target_net
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from models import Net

def test_attack_performance(dataloader, adv_GAN, target_model, dataset_size):
    n_correct = 0

    true_labels, pred_labels = [], []
    img_np, adv_img_np = [], []
    check_list = []
    for i, data in enumerate(dataloader, 0):
        img, true_label = data
        img, true_label = img.to(device), true_label.to(device)
        
        #targeted
        # true_label = torch.ones_like(true_label) * 0

        w = pretrained_G(img)
        adv_img = (torch.tanh(w / 15) + 1) / 2

        pred_label = torch.argmax(target_model(adv_img), 1)
        n_correct += torch.sum(pred_label == true_label, 0)

        true_labels.append(true_label.cpu().numpy())
        pred_labels.append(pred_label.cpu().numpy())


        img_np.append(img.detach().permute(0, 2, 3, 1).cpu().numpy())
        adv_img_np.append(adv_img.detach().permute(0, 2, 3, 1).cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    img_np = np.concatenate(img_np, axis=0)
    adv_img_np = np.concatenate(adv_img_np, axis=0)
    
    check_list = [true_labels != pred_labels]
    check_list = np.array(check_list).reshape(10000)
    
    only_correct = []
    only_correct_img = []
    only_correct_label = []

    for i in range(adv_img_np.shape[0]):
        if check_list[i] == True:
            only_correct.append(adv_img_np[i])
            only_correct_img.append(img_np[i])
            only_correct_label.append(true_labels[i])

    #Save data
    np.save('./npy/mlp_adv_img_np', np.array(only_correct))
    np.save('./npy/mlp_img_np', np.array(only_correct_img))
    np.save('./npy/mlp_label_np', np.array(only_correct_label))


    print('Correctly Classified: ', n_correct.item())
    print('Accuracy under attacks in {} set: {}%\n'.format('MNIST', 100 * n_correct.item()/dataset_size))

    for i, data in enumerate(dataloader, 0):
        img, true_label = data
        img, true_label = img.to(device), true_label.to(device)

        w = pretrained_G(img)
        adv_img = (torch.tanh(w / 15) + 1) / 2
        pred_label = torch.argmax(target_model(adv_img), 1)
        save_image(adv_img.data[:25], "./demo.png", nrow=5, normalize=True)
        print(pred_label[:25])

        break



if __name__ == '__main__':
    use_cuda=True
    image_nc=1
    batch_size = 128

    gen_input_nc = image_nc

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # load the pretrained model
    pretrained_model = "./MLP_target_model.pth"
    target_model = Net().to(device)
    target_model.load_state_dict(torch.load(pretrained_model))
    target_model.eval()

    # load the generator of adversarial examples
    pretrained_generator_path = './models/netG_epoch_100.pth'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # test adversarial examples in MNIST testing dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    

    test_attack_performance(dataloader=test_dataloader, adv_GAN=pretrained_G, target_model=target_model, dataset_size=len(mnist_dataset_test))
