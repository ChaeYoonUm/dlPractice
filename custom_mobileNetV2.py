import torch 
import torch.nn as nn

#========MobileNet V2========#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

num_classes = 30

class bottleNeckResidualBlock (nn.Module):
    # initialize
    def __init__(self, in_channels, out_channels, t, stride=1): # t = expansion factor
        super().__init__()
        
        #assert stride in [1,2]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias = False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace = True),
            
        )
        depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride = stride, padding = 1, groups = in_channels * t, bias = False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace = True),
        )
        pointwise = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1, bias = False),
            nn.BatchNorm2d(out_channels),
        )
        
        residual_list = []
        if t > 1:
            residual_list += [expand]
        residual_list += [depthwise, pointwise]
        self.residual = nn.Sequential(*residual_list)
    
    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            out = self.residual(x) + x
        else:
            out = self.residual(x)
    
        return out
              
class MobileNet_v2(nn.Module):
    def __init__(self, n_classes = num_classes):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace = True),
        )

        self.bottlenecks = nn.Sequential(
            self.make_stage(16, 8, t = 1, n = 1),
            self.make_stage(8, 12, t = 6, n = 2, stride = 1),
            self.make_stage(12, 16, t = 6, n = 3, stride = 2),
            self.make_stage(16, 32, t = 6, n = 4, stride = 1),
            self.make_stage(32, 68, t = 6, n = 3),
            self.make_stage(68, 80, t = 6, n = 3, stride = 2),
            self.make_stage(80, 160, t = 6, n = 1)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(160, 320, 1, bias = False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace = True),
            nn.Dropout(0.2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
        	nn.Dropout(0.2),
            nn.Linear(320, n_classes)
        )
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottlenecks(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x
    
    def make_stage(self, in_channels, out_channels, t, n, stride = 1):
        layers = [bottleNeckResidualBlock(in_channels, out_channels, t, stride)]
        in_channels = out_channels
        for _ in range(n):
            layers.append(bottleNeckResidualBlock(in_channels, out_channels, t))
        
        return nn.Sequential(*layers)
# # 모델 확인 
from torchinfo import summary
model = MobileNet_v2()
summary(model, (2,3,48,48), device="cpu")