import torch
import torch.nn as nn
import torchvision

class ResNet_Pose(nn.Module):
    def __init__(self, classes=22, allo=False):
        super(ResNet_Pose,self).__init__()
        self.classes = classes
        self.resnet = torchvision.models.resnet18()
        self.linear1 = nn.Linear(1000,2000)
        self.linear2 = nn.Linear(2000,2000)
        self.linear3 = nn.Linear(2000,4*classes)
        self. allo =  allo

    def forward(self, input):
        x = self.resnet(input)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x).view(input.shape[0],self.classes,4) 
        x[:,:,0] = torch.sigmoid(x[:,:,0])
        x = nn.functional.normalize(x, dim=2)
        
        return x

