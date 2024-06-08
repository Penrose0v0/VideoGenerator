import torch
import torch.nn as nn
import numpy as np

def get_attn_pad_mask(seq_q, seq_k):  
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    pad_attn_mask = seq_k[:,:,0].data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_causal_mask(seq):
    # seq: [batch_size, tgt_len, feature]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte() 
    return subsequence_mask 


class MultiHeadAttention(nn.Module): 
    def __init__(self, embed_dim, d_k, d_v, num_heads): 
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.w_q = nn.Linear(embed_dim, d_k * num_heads, bias=False)
        self.w_k = nn.Linear(embed_dim, d_k * num_heads, bias=False)
        self.w_v = nn.Linear(embed_dim, d_v * num_heads, bias=False)
        self.fc = nn.Linear(num_heads * d_v, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def ScaledDotProductAttention(self, q, k, v, attn_mask=None): 
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores.masked_fill_(attn_mask.bool(), -1e9) # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v) # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
    def forward(self, q, k, v, attn_mask=None): 
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = q, q.shape[0]
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        context, attn = self.ScaledDotProductAttention(q, k, v, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_v)

        output = self.dropout(self.fc(context))
        output = self.layer_norm(output + residual)
        return output, attn
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, embed_dim, bias=False))
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x): 
        residual = x
        output = self.dropout(self.fc(x))
        return self.layer_norm(output + residual) 
    

"""Transformer Decoder"""
class DecoderLayer(nn.Module): 
    def __init__(self, embed_dim, d_k, d_v, d_ff, num_heads): 
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, d_k, d_v, num_heads)
        self.ffn = PoswiseFeedForwardNet(embed_dim, d_ff)

    def forward(self, x, attn_mask=None): 
        y, attn = self.self_attention(x, x, x, attn_mask)
        y = self.ffn(y)
        return y, attn
    
class Decoder(nn.Module): 
    def __init__(self, num_layers, embed_dim, num_frames, num_tokens, 
                 d_k, d_v, d_ff, num_heads): 
        super().__init__()
        self.num_frames = num_frames  # Frame number per batch
        self.num_tokens = num_tokens  # Token number per frame
        self.embed_dim = embed_dim

        self.t_pos_emb = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.s_pos_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, d_k, d_v, d_ff, num_heads) for _ in range(num_layers)])
        
    def forward(self, x): 
        t_emb = self.t_pos_emb.repeat(1, 1, self.num_tokens).view(1, -1, self.embed_dim)
        s_emb = self.s_pos_emb.repeat(1, self.num_frames, 1)
        x = x + t_emb + s_emb

        # Masking
        casual_mask = get_causal_mask(x).to(x.device)
        pad_mask = get_attn_pad_mask(x, x).to(x.device)
        attn_mask = torch.gt((casual_mask + pad_mask), 0)

        attn_list = []
        for layer in self.layers: 
            x, attn = layer(x, attn_mask)
            attn_list.append(attn)
        return x, attn_list


"""World Model"""
class World_Model(nn.Module): 
    def __init__(self, vocab_size, embed_dim, num_layers, num_frames, num_tokens, 
                 d_k, d_v, d_ff, num_heads): 
        super().__init__()
        self.decoder = Decoder(num_layers, embed_dim, num_frames, num_tokens, d_k, d_v, d_ff, num_heads)
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x): 
        y, _ = self.decoder(x)
        y = self.projection(y)
        return nn.Softmax(dim=-1)(y)

if __name__ == "__main__": 
    dembed = 2048
    f = 12
    t = 576
    device = 'cpu'
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # net = MultiHeadAttention(4096, 16, 16, 64)
    net = World_Model(vocab_size=4096, embed_dim=dembed, num_layers=6, num_frames=f, num_tokens=t, 
                  d_k=64, d_v=64, d_ff=2048, num_heads=8).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    net = nn.DataParallel(net)

    inp = torch.rand(3, f * t, dembed).to(device)
    out = net(inp)
    print(out.shape)
    print(len(out[0, 0, :]), sum(out[0, 0, :]))