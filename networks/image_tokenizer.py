import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""2D UNet"""
class DoubleConv(nn.Sequential):
    def __init__(self, in_c, out_c, mid_c=None):
        if mid_c is None:
            mid_c = out_c
        super().__init__(
            nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )


class Down(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_c, out_c)
        )


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c // 2, in_c // 2, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv = DoubleConv(in_c, out_c, in_c // 2)

    def forward(self, cur, prev):
        cur = self.up(cur)
        cur = torch.cat([cur, prev], dim=1)
        return self.conv(cur)


class UNet(nn.Module):
    def __init__(self, in_c, out_c, base_c=64, num=4):
        super().__init__()
        self.in_conv = DoubleConv(in_c, base_c)  # 3->64
        
        down = [Down(base_c * (2 ** i), base_c * (2 ** (i + 1))) for i in range(num - 1)]
        down.append(Down(base_c * (2 ** (num - 1)), base_c * (2 ** (num - 1))))
        self.down = nn.ModuleList(down)

        up = [Up(base_c * (2 ** (i + 1)), base_c * (2 ** (i - 1))) for i in range(num - 1, 0, -1)]
        up.append(Up(base_c * 2, base_c))
        self.up = nn.ModuleList(up)

        self.out_conv = nn.Conv2d(base_c, out_c, kernel_size=1)

    def encode(self, x):
        cur = self.in_conv(x)
        prev_list = []

        for down in self.down:
            prev_list.append(cur)
            cur = down(cur)

        return cur, prev_list

    def decode(self, cur, prev_list):
        for prev, up in zip(prev_list[::-1], self.up):
            cur = up(cur, prev)

        return self.out_conv(cur)

    def forward(self, x):
        return self.decode(*self.encode(x))
    

"""Vector Quantizer"""
class VectorQuantizerEMA(nn.Module):
    def __init__(self, vocab_size, embed_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
 
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
 
        self._embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
 
        self.register_buffer('_ema_cluster_size', torch.zeros(vocab_size))
        self._ema_w = nn.Parameter(torch.Tensor(vocab_size, self.embed_dim))
        self._ema_w.data.normal_()
 
        self._decay = decay
        self._epsilon = epsilon
 
    def forward(self, inputs):
        flat_input = inputs
        
        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
 
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.vocab_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
 
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight)
 
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
 
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self.vocab_size * self._epsilon) * n)
 
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
 
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
 
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
 
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices
    

"""Discriminator"""
class Discriminator(nn.Module):
    def __init__(self, patch=True):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_c, out_c, bn=True):
            block = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if bn:
                block.append(nn.BatchNorm2d(out_c))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.layers = nn.Sequential(
            *discriminator_block(3, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        self.sigmoid = nn.Sigmoid()

        self.patch = patch
        if not patch:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1)
        else:
            self.conv = nn.Conv2d(512, 1, kernel_size=1, bias=False)

    
    def forward(self, x):
        x = self.layers(x)

        if not self.patch:
            x = self.avgpool(x)
            x = self.fc(torch.flatten(x, 1))
        else: 
            x = self.conv(x)

        return self.sigmoid(x)
    
    
"""Image Tokenizer"""
class Image_Tokenizer(UNet):
    def __init__(self, in_c=3, out_c=3, base_c=64, num=4, 
                 vocab_size=8192, embed_dim=4096, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__(in_c, out_c, base_c, num) 
        # Quantization loss
        self.etoken = nn.Conv2d(base_c * (2 ** (num - 1)), embed_dim, kernel_size=1, bias=False)
        self.vq = VectorQuantizerEMA(vocab_size, embed_dim, commitment_cost, decay, epsilon)
        self.dtoken = nn.Conv2d(embed_dim, base_c * (2 ** (num - 1)), kernel_size=1, bias=False)
        self.embed_dim = embed_dim

        # Perceptual loss
        percept = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.percept = torch.nn.Sequential(*percept.children())[:-2]
        for param in self.percept.parameters(): 
            param.requires_grad = False

        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.dim_align = nn.Conv2d(512, 384, kernel_size=1, bias=False)
    
    def tokenize(self, x): 
        cur, _ = super().encode(x)
        z = self.etoken(cur)

        # Convert z from BCHW -> BHWC and then flatten it
        z = z.permute(0, 2, 3, 1).contiguous()
        batch_size = z.shape[0]
        z = z.view(-1, self.embed_dim)

        # Tokenize
        _, tokens, _, indices = self.vq(z)
        tokens = tokens.view(batch_size, -1, self.embed_dim)

        return tokens, indices

    def quantize(self, cur):
        z = self.etoken(cur)

        # Convert z from BCHW -> BHWC and then flatten it
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        z = z.view(-1, self.embed_dim)

        # Discretize z
        vq_loss, tokens, _, _ = self.vq(z)

        # Unflatten z and convert it from BHWC -> BCHW
        z = tokens.view(z_shape)
        z = z.permute(0, 3, 1, 2).contiguous()

        cur = self.dtoken(z)
        return cur, vq_loss

    def generate(self, x): 
        cur, prev_list = self.encode(x)
        cur, _ = self.quantize(cur)
        return self.decode(cur, prev_list)
    
    def calculate_loss(self, x): 
        # Perceptual loss
        cur, prev_list = self.encode(x)
        per = self.percept(x)
        perceptual_loss = F.mse_loss(F.max_pool2d(cur, kernel_size=2), per)

        # VQ loss
        cur, vq_loss = self.quantize(cur)

        # Recon loss
        y = self.decode(cur, prev_list)
        l2_loss = F.mse_loss(y, x)
        l1_loss = F.l1_loss(y, x)

        # Inductive bias loss
        dino = self.dino.get_intermediate_layers(x, n=1)[0][:, 1:, :]
        base = self.dim_align(cur).view(cur.shape[0], 384, -1).permute(0, 2, 1)
        inductive_bias_loss = F.cosine_similarity(base, dino, dim=-1).mean()

        # Generate output
        output = self.decode(cur, prev_list)

        return perceptual_loss, vq_loss, l1_loss, l2_loss, inductive_bias_loss, output
    
    def forward(self, x): 
        return self.calculate_loss(x)


"""New Image Tokenizer"""
class new_Image_Tokenizer(nn.Module): 
    def __init__(self, vocab_size, embed_dim, 
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5): 
        super().__init__()
        self.hidden = 1024
        self.embed_dim = embed_dim

        self.encoder = torch.nn.Sequential(*models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).children())[:-3]  # [batch_size, 1024, 18, 32]
        self.etoken = nn.Conv2d(self.hidden, embed_dim, kernel_size=1, bias=False)

        self.vq = VectorQuantizerEMA(vocab_size, embed_dim, commitment_cost, decay, epsilon)

        self.dtoken = nn.Conv2d(embed_dim, self.hidden, kernel_size=1, bias=False)
        decoder_channels = [512, 256, 128, 64]
        decoder_kernel = [4, 4, 4, 4]
        self.decoder = self.make_decoder_layer(decoder_channels, decoder_kernel)

    def make_decoder_layer(self, num_channels, num_kernel):
        layers = []
        prev_channels = self.hidden
        for channels, kernel in zip(num_channels, num_kernel):
            up = nn.ConvTranspose2d(
                in_channels=prev_channels,
                out_channels=channels,
                kernel_size=kernel,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False)

            layers.append(up)
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = channels

        layers.append(
            nn.Sequential(
                nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(prev_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(prev_channels, 3, kernel_size=1, stride=1, padding=0, bias=True),
            )
        )

        return nn.Sequential(*layers)
    
    def tokenize(self, x): 
        cur = self.encoder(x)
        z = self.etoken(cur)

        # Convert z from BCHW -> BHWC and then flatten it
        z = z.permute(0, 2, 3, 1).contiguous()
        batch_size = z.shape[0]
        z = z.view(-1, self.embed_dim)

        # Tokenize
        _, tokens, _, indices = self.vq(z)
        tokens = tokens.view(batch_size, -1, self.embed_dim)

        return tokens, indices
    
    def generate(self, x):
        cur = self.encoder(x)
        z = self.etoken(cur)

        # Convert z from BCHW -> BHWC and then flatten it
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        z = z.view(-1, self.embed_dim)

        # Discretize z
        vq_loss, tokens, _, _ = self.vq(z)

        # Unflatten z and convert it from BHWC -> BCHW
        z = tokens.view(z_shape)
        z = z.permute(0, 3, 1, 2).contiguous()

        cur = self.dtoken(z)
        y = self.decoder(cur)
        
        return y
    
    def forward(self, x): 
        cur = self.encoder(x)
        z = self.etoken(cur)

        # Convert z from BCHW -> BHWC and then flatten it
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        z = z.view(-1, self.embed_dim)

        # Discretize z
        vq_loss, tokens, _, _ = self.vq(z)

        # Unflatten z and convert it from BHWC -> BCHW
        z = tokens.view(z_shape)
        z = z.permute(0, 3, 1, 2).contiguous()

        cur = self.dtoken(z)
        y = self.decoder(cur)

        # Loss
        perceptual_loss = torch.zeros_like(vq_loss)
        l2_loss = F.mse_loss(y, x)
        l1_loss = F.l1_loss(y, x)

        return perceptual_loss, vq_loss, l1_loss, l2_loss, y

if __name__ == "__main__": 
    # net = UNet(in_c=3, out_c=3, num=4)
    # net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
    # net = torch.nn.Sequential(*net.children())[:-2]
    # net = Discriminator(patch=False)
    # net = Image_Tokenizer(embed_dim=4096)
    # net = torch.nn.Sequential(*models.vgg16().children())[:-2]
    # net = Image_Tokenizer(vocab_size=4096, embed_dim=2048)
    # net = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    net = new_Image_Tokenizer(vocab_size=4096, embed_dim=2048)

    inp = torch.rand(1, 3, 288, 512)
    out = net(inp)
    # print(out.shape)
