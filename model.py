import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = torch.nn.ModuleList(modules=[
            # Input is (28, 28, 1) = 784.
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), stride=1, padding='valid'),
            # Representation here will be (26, 26, 4) = 2,704.
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3,3), stride=2, padding='valid'),
            # Representation here will be (12, 12, 16) = 2,304.
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding='valid'),
            # Representation here will be (5, 5, 32) = 800.
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding='valid'),
            # Representation here will be (3, 3, 32) = 576.
        ])
        self.fc_layer = torch.nn.Linear(in_features=576, out_features=10)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_tensor: torch.Tensor):
        
        hidden_layer_repr = input_tensor
        for conv_layer in self.conv_stack:
            hidden_layer_repr = conv_layer(hidden_layer_repr)
        
        # Also known as the pre-activations (i.e. values before an activation function).
        logits = self.fc_layer(hidden_layer_repr)
        # It's more common to use a softmax activation for a multi-class classification, but I'm intentionally
        # choosing a per-class sigmoid instead to allow this architecture to easily generalize to multi-label
        # scenarios. In other words, cases where an input can belong to more than one output class.
        per_class_probabilites = self.sigmoid(logits)
        
        return per_class_probabilites
        