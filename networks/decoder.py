import torch
import torch.nn as nn
import torch.nn.functional as F

from image_tokenizer import Image_Tokenizer

class Decoder(nn.Module): 
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        decoder_channels = [512, 256, 128, 64]
        decoder_kernel = [4, 4, 4, 4]
        self.decoder = self.make_decoder_layer(decoder_channels, decoder_kernel)

        self.image_tokenizer = Image_Tokenizer(vocab_size=vocab_size, embed_dim=embed_dim)
        self.image_encoder = self.image_tokenizer.tokenize
        self.dtoken = self.image_tokenizer.dtoken

        for param in self.image_tokenizer.parameters(): 
            param.requires_grad = False
        
    def load_image_tokenizer(self, it_model_path): 
        self.image_tokenizer.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(it_model_path).items()})

    def make_decoder_layer(self, num_channels, num_kernel):
        layers = []
        prev_channels = 512
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

    def forward(self, x): 
        cur, _ = self.image_tokenizer.encode(x)
        cur, _ = self.image_tokenizer.quantize(cur)
        return self.decoder(cur)
    
    def decode_tokens(self, tokens): 
        z = self.dtoken(tokens)
        return self.decoder(z)

if __name__ == "__main__": 
    net = Decoder(4096, 2048)
    inp = torch.rand(5, 2048, 18, 32)
    out = net.decode_tokens(inp)
    print(out.shape)