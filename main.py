import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import MNIST_target_net
from models import Net
from WGAN import WGAN_Attack
import os

if __name__ == "__main__":
    use_cuda=True
    image_nc=1
    epochs = 100
    batch_size = 128
    BOX_MIN = 0
    BOX_MAX = 1

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    #Load classfier
    # pretrained_model = "./MNIST_target_model_origin.pth"
    pretrained_model = "./MLP_target_model.pth"
    # targeted_model = MNIST_target_net().to(device)
    targeted_model = Net().to(device)
    targeted_model.load_state_dict(torch.load(pretrained_model))
    targeted_model.eval()
    model_num_labels = 10

    # MNIST train dataset and dataloader declaration
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    WGAN = WGAN_Attack(device,
                        targeted_model,
                        model_num_labels,
                        image_nc,
                        BOX_MIN,
                        BOX_MAX)
                            
    isExists = os.path.exists('./models/')
    if isExists == False:
        os.mkdir('./models/')
        
    WGAN.train(dataloader, epochs)
