import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks.image_tokenizer import Image_Tokenizer

"""Action Encoder"""
class Action_Encoder(nn.Module): 
    def __init__(self, action_space_dim=25, embed_dim=4096, num_action_tokens=25): 
        super().__init__()
        self.fc = nn.Linear(action_space_dim, embed_dim * num_action_tokens)
        self.embed_dim = embed_dim
        self.num_action_tokens = num_action_tokens
    
    def forward(self, x): 
        return self.fc(x).view(-1, self.num_action_tokens, self.embed_dim)
    

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
        seq_len = x.shape[1]
        t_emb = self.t_pos_emb.repeat(1, 1, self.num_tokens).view(1, -1, self.embed_dim)[:, :seq_len, :]
        s_emb = self.s_pos_emb.repeat(1, self.num_frames, 1)[:, :seq_len, :]
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


"""Transformer"""
class Transformer(nn.Module): 
    def __init__(self, vocab_size, embed_dim, num_frames, num_tokens, num_layers=6, 
                 d_k=512, d_v=512, d_ff=2048, num_heads=8): 
        super().__init__()
        self.decoder = Decoder(num_layers, embed_dim, num_frames, num_tokens, d_k, d_v, d_ff, num_heads)
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x): 
        y, _ = self.decoder(x)
        y = self.projection(y)
        # return nn.Softmax(dim=-1)(y)
        return y
    

"""World Model"""
class World_Model(nn.Module): 
    def __init__(self, vocab_size, embed_dim, num_frames, num_image_tokens, num_action_tokens): 
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_image_tokens = num_image_tokens
        self.num_action_tokens = num_action_tokens

        self.image_tokenizer = Image_Tokenizer(vocab_size=vocab_size, embed_dim=embed_dim)
        self.image_encoder = self.image_tokenizer.tokenize
        self.action_encoder = Action_Encoder(embed_dim=embed_dim, num_action_tokens=num_action_tokens)
        self.transformer = Transformer(vocab_size=vocab_size, embed_dim=embed_dim, num_frames=num_frames, num_tokens=num_image_tokens+num_action_tokens)
        
        for param in self.image_tokenizer.parameters(): 
            param.requires_grad = False
    
    def load_image_tokenizer(self, it_model_path): 
        self.image_tokenizer.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(it_model_path).items()})

    def forward(self, frame_list, action_list):
        """
        frame_list: [batch_size, num_frames, c, h, w]
        action_list: [batch_size, num_frames-1, action_size]
        """ 
        # Encode frames and actions
        batch_size, _, c, h, w = frame_list.shape
        frames = frame_list.view(-1, c, h, w)
        encoded_frames, indices = self.image_encoder(frames)
        encoded_frames = encoded_frames.view(
            batch_size, self.num_frames, self.num_image_tokens, self.embed_dim)

        action_size = action_list.shape[-1]
        actions = action_list.view(-1, action_size)
        encoded_actions = self.action_encoder(actions).view(
            batch_size, self.num_frames - 1, self.num_action_tokens, self.embed_dim)
        tmp_action_tokens = F.pad(encoded_actions, (0, 0, 0, 0, 0, 1))

        input_tokens = torch.cat((encoded_frames, tmp_action_tokens), dim=2).view(batch_size, -1, self.embed_dim)

        # Process output
        output = self.transformer(input_tokens).view(batch_size, self.num_frames, -1, self.vocab_size)
        output = output[:, :, :-self.num_action_tokens, :].reshape(-1, self.vocab_size)
        target = indices.squeeze()
        loss = F.cross_entropy(output, target)
        return loss
    
    def predict(self, frame_list, action_list): 
        """
        frame_list: [1, len_f, c, h, w]
        next_action: [1, len_a, action_size]
        p.s. len_a >= len_f
        """ 
        max_len = self.num_frames * (self.num_image_tokens + self.num_action_tokens)
        pred_times = action_list.shape[1] - frame_list.shape[1] + 1
        assert pred_times > 0

        # Encode frames and actions
        batch_size, cur_frame_num, c, h, w = frame_list.shape
        frames = frame_list.view(-1, c, h, w)
        encoded_frames, _ = self.image_encoder(frames)
        encoded_frames = encoded_frames.view(
            batch_size, cur_frame_num, self.num_image_tokens, self.embed_dim)

        _, total_action_num, action_size = action_list.shape
        actions = action_list.view(-1, action_size)
        encoded_actions = self.action_encoder(actions).view(
            batch_size, total_action_num, self.num_action_tokens, self.embed_dim)
        former_actions, latter_actions = encoded_actions[:, :cur_frame_num], encoded_actions[:, cur_frame_num:]

        input_tokens = torch.cat((encoded_frames, former_actions), dim=2).view(batch_size, -1, self.embed_dim)
        cur_token_len = input_tokens.size(1)

        # print(encoded_frames.shape, encoded_actions.shape)
        # print(input_tokens.shape)

        generated_tokens_list = []
        for i in range(pred_times): 
            if i != 0:
                input_tokens = torch.cat((input_tokens, latter_actions[:, i-1]), dim=1) 
            for j in range(self.num_image_tokens): 
                _, output = self.transformer(input_tokens).max(dim=-1, keepdim=False)
                next_token_indice = output[:, -1]
                next_token = self.image_tokenizer.vq._embedding.weight[next_token_indice].unsqueeze(1)
                input_tokens = torch.cat((input_tokens, next_token), dim=1)
                print(f'\r{j}', end='')
            generated_tokens = input_tokens[:, -self.num_image_tokens:, :]
            generated_tokens = generated_tokens.reshape(batch_size, 18, 32, self.embed_dim).permute(0, 3, 1, 2).contiguous()
            # print(generated_tokens.shape)
            generated_tokens_list.append(generated_tokens)
        # generated_tokens = torch.rand(1, 576, 2048).to(device)
        # z = generated_tokens.view(batch_size, 18, 32, -1)
        # z = z.permute(0, 3, 1, 2).contiguous()
        return generated_tokens_list

    def predict_from_one_frame(self, frame, action_list): 
        """
        frame: [batch_size, c, h, w]
        action_list: [batch_size, len, action_size]
        """
        cur, prev_list = self.image_tokenizer.encode(frame)
        z = self.image_tokenizer.etoken(cur)

        # Convert z from BCHW -> BHWC and then flatten it
        z = z.permute(0, 2, 3, 1).contiguous()
        batch_size = z.shape[0]
        z = z.view(-1, self.embed_dim)

        # Tokenize the frame
        _, tokens, _, indices = self.image_tokenizer.vq(z)
        input_tokens = tokens.view(batch_size, -1, self.embed_dim)
        print(tokens.shape)

        # Tokenize actions
        _, total_action_num, action_size = action_list.shape
        actions = action_list.view(-1, action_size)
        encoded_actions = self.action_encoder(actions).view(
            batch_size, total_action_num, self.num_action_tokens, self.embed_dim)
        print(encoded_actions.shape)

        # Predicting
        generated_tokens_list = []
        for i in range(total_action_num): 
            input_tokens = torch.cat((input_tokens, encoded_actions[:, i]), dim=1)
            for j in range(self.num_image_tokens): 
                _, output = self.transformer(input_tokens).max(dim=-1, keepdim=False)
                next_token_indice = output[:, -1]
                next_token = self.image_tokenizer.vq._embedding.weight[next_token_indice].unsqueeze(1)
                input_tokens = torch.cat((input_tokens, next_token), dim=1)
                print(f'\r{j}', end='')
            generated_tokens = input_tokens[:, -self.num_image_tokens:, :]
            generated_tokens = generated_tokens.reshape(batch_size, 18, 32, self.embed_dim).permute(0, 3, 1, 2).contiguous()
            generated_tokens_list.append(generated_tokens)

        decoded_list = []
        for generated_tokens in generated_tokens_list: 
            cur = self.image_tokenizer.dtoken(generated_tokens)
            decoded = self.image_tokenizer.decode(cur, prev_list)
            print(decoded.shape)
            decoded_list.append(decoded)

        return decoded_list


if __name__ == "__main__": 
    dembed = 2048
    f = 7  # num of frame
    t = 576 + 25  # tokens per frame  32 * 18 + action tokens 4
    device = 'cpu'
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # net = MultiHeadAttention(4096, 16, 16, 64)
    # net = Transformer(vocab_size=4096, embed_dim=dembed, num_layers=6, num_frames=f, num_tokens=t, 
    #               d_k=2048, d_v=2048, d_ff=2048, num_heads=8).to(device)
    net = World_Model(vocab_size=4096, embed_dim=dembed, num_frames=f, num_image_tokens=576, num_action_tokens=25).to(device)
    # total_params = sum(p.numel() for p in net.parameters())
    # print(total_params)
    # net = nn.DataParallel(net)

    # inp = torch.rand(1, f * t, dembed).to(device)
    fl = torch.rand(1, 3, 288, 512).to(device)
    al = torch.rand(1, 3, 25).to(device)
    out = net.predict_from_one_frame(fl, al)
    # for o in out: 
    #     print(o.shape)
    # print(out.shape)
    # print(len(out[0, 0, :]), sum(out[0, 0, :]))