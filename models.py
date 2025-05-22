import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN2(nn.Module):
    def __init__(self, in_channels=3, input_size=32, n_kernels=500, out_dim=1):
        super(SimpleCNN2, self).__init__()
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(in_channels, n_kernels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(n_kernels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(2 * n_kernels)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Define self.pool here
        
        # Calculate output size after convolutions
        conv1_out_size = (input_size - 5 + 2 * 2) // 1 + 1
        pool1_out_size = conv1_out_size // 2
        conv2_out_size = (pool1_out_size - 5) // 1 + 1
        pool2_out_size = conv2_out_size // 2

        flattened_size = 2 * n_kernels * pool2_out_size * pool2_out_size

        # Fully connected layers
        self.latent_dim = 120
        self.fc1 = nn.Linear(flattened_size, self.latent_dim)
        self.bn3 = nn.BatchNorm1d(self.latent_dim)
        self.classifier = nn.Linear(self.latent_dim, out_dim)

        self.size = input_size
        self.channels = in_channels
        self.n_kernels = n_kernels
        self.out_dim = out_dim


    def produce_feature(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        return x

    def forward(self, x):
        x = self.produce_feature(x)
        out = self.classifier(x)
        return out
    

    def freeze_feature_extractor(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.bn3.parameters():
            param.requires_grad = False
        print("Feature extractor frozen.")

    def unfreeze_feature_extractor(self):
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.bn1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True
        for param in self.bn2.parameters():
            param.requires_grad = True
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.bn3.parameters():
            param.requires_grad = True
        print("Feature extractor unfrozen.")

    def freeze_conv_layers(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False 
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        print("Conv layers frozen.")

    def unfreeze_conv_layers(self):
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.bn1.parameters():
            param.requires_grad = True  
        for param in self.conv2.parameters():
            param.requires_grad = True
        for param in self.bn2.parameters():
            param.requires_grad = True
        print("Conv layers unfrozen.")

    def freeze_all_layers(self):
        self.freeze_feature_extractor()
        for param in self.classifier.parameters():
            param.requires_grad = False
        print("All layers frozen.")

    def unfreeze_all_layers(self):
        self.unfreeze_feature_extractor()
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("All layers unfrozen.")

class SimpleCNN2_MOON(nn.Module):
    def __init__(self, in_channels=3, input_size=32, n_kernels=500, out_dim=10, proj_dim=128):
        super(SimpleCNN2_MOON, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_kernels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(n_kernels)

        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(2 * n_kernels)

        self.pool = nn.MaxPool2d(2, 2)

        conv1_out_size = (input_size - 5 + 2 * 2) // 1 + 1
        pool1_out_size = conv1_out_size // 2
        conv2_out_size = (pool1_out_size - 5) // 1 + 1
        pool2_out_size = conv2_out_size // 2
        flattened_size = 2 * n_kernels * pool2_out_size * pool2_out_size

        self.latent_dim = 120
        self.fc1 = nn.Linear(flattened_size, self.latent_dim)
        self.bn3 = nn.BatchNorm1d(self.latent_dim)

        # Projection head (MOON part)
        self.proj1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.proj2 = nn.Linear(self.latent_dim, proj_dim)

        # Final classifier
        self.classifier = nn.Linear(proj_dim, out_dim)

    def feature_extractor(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        return x  # this is h

    def forward(self, x):
        h = self.feature_extractor(x)
        proj = F.relu(self.proj1(h))
        proj = self.proj2(proj)
        y = self.classifier(proj)
        return h, proj, y

class SimpleCNN2Encoder(nn.Module):
    def __init__(self, in_channels=3, input_size=32, n_kernels=500):
        super(SimpleCNN2Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_kernels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(n_kernels)

        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(2 * n_kernels)
        self.pool = nn.MaxPool2d(2, 2)

        conv1_out_size = (input_size - 5 + 2 * 2) // 1 + 1
        pool1_out_size = conv1_out_size // 2
        conv2_out_size = (pool1_out_size - 5) // 1 + 1
        pool2_out_size = conv2_out_size // 2

        self.flattened_size = 2 * n_kernels * pool2_out_size * pool2_out_size

        self.fc1 = nn.Linear(self.flattened_size, 120)
        self.bn3 = nn.BatchNorm1d(120)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        return x
    
class SimpleCNN2Classifier(nn.Module):
    def __init__(self, in_dim=120, out_dim=10):  # or out_dim=1 for binary
        super(SimpleCNN2Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)




class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=64, img_size=32):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, 
                 embed_dim=64, depth=4, num_heads=2, mlp_ratio=2.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True  # Ensures input is (B, N, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.size = img_size
        self.channels = in_channels
        self.out_dim = num_classes

    def produce_feature(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)  # (B, N+1, D)
        return self.norm(x[:, 0])  # Return CLS token

    def forward(self, x):
        x = self.produce_feature(x)
        return self.head(x)
