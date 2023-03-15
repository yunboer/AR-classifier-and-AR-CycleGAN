import torch
import torch.nn as nn
import torch.nn.functional as F

class AANet_weight(nn.Module):
    def __init__(self, args, num_classes=3):
        super().__init__()
        self.ncolorspace = args.ncolorspace
        self.num_classes = num_classes
        self.nlayer = args.nlayer  # num of layers
        self.switch = args.switch  # add or concat

        self._make_branch(args.ncolorspace, num_classes=self.num_classes, nlayer=self.nlayer)
        self.backBranch = AAbackBranch(num_classes=self.num_classes, nlayer=self.nlayer, switch=self.switch,
                                     ncolorspace=self.ncolorspace)

        self.dropout = nn.Dropout(p=0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)

    def _make_branch(self, ncs, num_classes=3, nlayer=2):
        shape_list = [
            (3, 224, 224), 
            (64, 224, 224), 
            (128, 112, 112), 
            (256, 56, 56), 
            (512, 28, 28), 
            (3, 3)]

        self.preBranch1 = AApreBranch(num_classes=num_classes, nlayer=nlayer) if ncs >= 1 else None
        self.preBranch2 = AApreBranch(num_classes=num_classes, nlayer=nlayer) if ncs >= 2 else None
        self.preBranch3 = AApreBranch(num_classes=num_classes, nlayer=nlayer) if ncs >= 3 else None
        self.preBranch4 = AApreBranch(num_classes=num_classes, nlayer=nlayer) if ncs >= 4 else None
        
        if self.switch == 'add':
            self.weight1 = torch.nn.Parameter(
                torch.rand((1), dtype=torch.float32, requires_grad=True)) if ncs >= 1 else None
            self.weight2 = torch.nn.Parameter(
                torch.rand((1), dtype=torch.float32, requires_grad=True)) if ncs >= 2 else None
            self.weight3 = torch.nn.Parameter(
                torch.rand((1), dtype=torch.float32, requires_grad=True)) if ncs >= 3 else None
            self.weight4 = torch.nn.Parameter(
                torch.rand((1), dtype=torch.float32, requires_grad=True)) if ncs >= 4 else None


    def forward(self, x) -> torch.Tensor:
        if self.switch == 'add':
            if self.ncolorspace == 1:
                out = self.preBranch1(x[0]) * self.weight1 / self.weight1
            if self.ncolorspace == 2:
                out = (self.preBranch1(x[0]) * self.weight1 + \
                       self.preBranch2(x[1]) * self.weight2) / \
                      (self.weight1 + self.weight2)
            if self.ncolorspace == 3:
                out = (self.preBranch1(x[0]) * self.weight1 + \
                       self.preBranch2(x[1]) * self.weight2 + \
                       self.preBranch3(x[2]) * self.weight3) / \
                      (self.weight1 + self.weight2 + self.weight3)
            if self.ncolorspace == 4:
                out = (self.preBranch1(x[0]) * self.weight1 + \
                       self.preBranch2(x[1]) * self.weight2 + \
                       self.preBranch3(x[2]) * self.weight3 + \
                       self.preBranch3(x[3]) * self.weight4) / \
                      (self.weight1 + self.weight2 + self.weight3 + self.weight4)

        else:# concat
            if self.ncolorspace == 1:
                out = self.preBranch1(x[0])
            if self.ncolorspace == 2:
                out = torch.concat((self.preBranch1(x[0]),
                                    self.preBranch2(x[1])), dim=1)
            if self.ncolorspace == 3:
                out = torch.concat((self.preBranch1(x[0]),
                                    self.preBranch2(x[1]),
                                    self.preBranch3(x[2])), dim=1)
            if self.ncolorspace == 4:
                out = torch.concat((self.preBranch1(x[0]),
                                    self.preBranch2(x[1]),
                                    self.preBranch3(x[2]),
                                    self.preBranch4(x[3])), dim=1)
                
        out = self.backBranch(out)
        out = self.dropout(out)
        return out
    
class AAbackBranch(nn.Module):
    def __init__(self, num_classes=3, nlayer=2, switch='add', ncolorspace=2) -> None:
        super().__init__()
        block = BasicBlock

        self.num_classes = num_classes
        self.nlayer = nlayer

        k = ncolorspace
        if switch == 'add':
            k = 1

        layer_list = []
        if self.nlayer <= 0:
            k = 1 if nlayer != 0 else k
            self.Conv = nn.Conv2d(3 * k, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.BN = nn.BatchNorm2d(64)
            layer_list.append(self._make_layer(block, in_planes=64, planes=64, stride=1))
        if self.nlayer <= 1:
            k = 1 if nlayer != 1 else k
            layer_list.append(self._make_layer(block, in_planes=64 * k, planes=128, stride=2))
        if self.nlayer <= 2:
            k = 1 if nlayer != 2 else k
            layer_list.append(self._make_layer(block, in_planes=128 * k, planes=256, stride=2))
        if self.nlayer <= 3:
            k = 1 if nlayer != 3 else k
            layer_list.append(self._make_layer(block, in_planes=256 * k, planes=512, stride=2))
        if self.nlayer <= 4:
            k = 1 if nlayer != 4 else k
            layer_list.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.linear = nn.Linear(512 * block.expansion * k, self.num_classes)
        if self.nlayer == 5:
            layer_list.append(nn.Linear(self.num_classes * k, self.num_classes))
        self.pre_layers = nn.Sequential(*layer_list)

    def _make_layer(self, block, in_planes, planes, stride):
        return nn.Sequential(
            block(in_planes, planes, stride, False),
            block(planes, planes, 1, True)
        )

    def forward(self, x):
        out = x
        if self.nlayer == 0:
            out = F.relu(self.BN(self.Conv(out)), inplace=False)
            out = self.pre_layers(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.nlayer < 5:
            out = self.pre_layers(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        return out
    
    
class AApreBranch(nn.Module):
    def __init__(self, num_classes=3, nlayer=2) -> None:
        super().__init__()
        block = BasicBlock

        self.num_classes = num_classes
        self.nlayer = nlayer

        layer_list = []
        if self.nlayer >= 1:
            self.Conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.BN = nn.BatchNorm2d(64)
            layer_list.append(self._make_layer(block, in_planes=64, planes=64, stride=1))
        if self.nlayer >= 2:
            layer_list.append(self._make_layer(block, in_planes=64, planes=128, stride=2))
        if self.nlayer >= 3:
            layer_list.append(self._make_layer(block, in_planes=128, planes=256, stride=2))
        if self.nlayer >= 4:
            layer_list.append(self._make_layer(block, in_planes=256, planes=512, stride=2))
        if self.nlayer >= 5:
            layer_list.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.linear = nn.Linear(512 * block.expansion, self.num_classes)
        if self.nlayer >= 1:
            self.pre_layers = nn.Sequential(*layer_list)

    def _make_layer(self, block, in_planes, planes, stride):
        return nn.Sequential(
            block(in_planes, planes, stride, False),
            block(planes, planes, 1, True)
        )

    def forward(self, x):
        out = x
        if self.nlayer >= 1:
            out = F.relu(self.BN(self.Conv(out)), inplace=False)
            out = self.pre_layers(out)
        if self.nlayer == 5:
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out1 = F.relu(out,inplace=False)
        return out1
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--ncolorspace', type=int, default=3) 
    parser.add_argument('--model', type=str, default='AANet') 
    parser.add_argument('--num_classes',type=int,default=3)
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--switch', type=str, default='add', choices=['add', 'concat'])
    
    arguements = parser.parse_args()
    
    model = AANet_weight(arguements)
    
    input = [torch.rand((4,3,224,224)) for i in range(3)]
    
    output = model(input)
    
    # print(output[])
    for item in output:
        print(item.shape)



