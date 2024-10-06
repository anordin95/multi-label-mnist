import torch

class ConvStackModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_stack = torch.nn.ModuleList(modules=[
            # Note: PyTorch generally expects the channel-dimension first, i.e. (C,H,W) not (H,W,C).
            # Input representation is (1, 28, 28) = 784. That is, grayscale (so 1 channel) and 28x28 pixels.
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), stride=1, padding='valid'),
            # Representation here will have dimensions: (4, 26, 26) = 2,704.
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3,3), stride=2, padding='valid'),
            # Representation here will have dimensions: (16, 12, 12) = 2,304.
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding='valid'),
            # Representation here will have dimensions: (32, 5, 5) = 800.
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding='valid'),
            # Representation here will have dimensions: (32, 3, 3) = 576.
        ])
        self.fc_layer = torch.nn.Linear(in_features=576, out_features=10)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_tensor: torch.Tensor):

        hidden_layer_repr = input_tensor
        for conv_layer in self.conv_stack:
            hidden_layer_repr = conv_layer(hidden_layer_repr)
        
        if input_tensor.shape[0] == 1:
            # The batch-dimension is either not present or is 1, so it's safe to flatten entirely.
            flattened_hidden_layer_repr = hidden_layer_repr.flatten(start_dim=0)
        else:
            # Flatten all dimensions except for the batch-dimension.
            flattened_hidden_layer_repr = hidden_layer_repr.flatten(start_dim=1)

        # Also known as the pre-activations (i.e. values before an activation function).
        logits = self.fc_layer(flattened_hidden_layer_repr)
        
        # It's more common to use a softmax activation for a multi-class classification, but I'm intentionally
        # choosing a per-class sigmoid instead to allow this architecture to easily generalize to multi-label
        # scenarios. In other words, cases where an input can belong to more than one output class.
        per_class_probabilites = self.sigmoid(logits)
        
        return per_class_probabilites
        