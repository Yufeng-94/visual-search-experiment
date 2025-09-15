import torch

class PreTrainedImageEncoder(torch.nn.Module):
    def __init__(
            self, 
            pre_trained_model: torch.nn.Module, 
            device: torch.device=torch.device('cpu')
            ):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            dict(pre_trained_model.named_children())
        ).to(device)

        # Remove 'avgpool' and 'fc' layers
        if 'avgpool' in self.layers:
            del self.layers['avgpool']
        if 'fc' in self.layers:
            del self.layers['fc']

        # Add GAP layer
        self.layers['gap'] = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers['conv1'](x)
        x = self.layers['bn1'](x)
        x = self.layers['relu'](x)
        x = self.layers['maxpool'](x)
        x = self.layers['layer1'](x)
        x = self.layers['layer2'](x)
        x = self.layers['layer3'](x)
        x = self.layers['layer4'](x)
        x = self.layers['gap'](x) # [batch_size, num_channels, 1, 1]
        x = x.flatten(1) # [batch_size, num_channels]

        return x
    
