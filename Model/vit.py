import torch
from torch import nn, einsum
from einops import rearrange, repeat

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BatchNorm = nn.BatchNorm1d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.attn_gradients = None
        self.attention_map = None
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
    def get_attn_gradients(self):
        return self.attn_gradients
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
    def get_attention_map(self):
        return self.attention_map
    def forward(self, x, register_hook=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_dim, dropout=dropout, act_layer=act_layer)
    def forward(self, x, register_hook=False):
        x = x + self.attn(self.norm1(x), register_hook=register_hook)
        x = x + self.mlp(self.norm2(x))
        return x
class ViT(nn.Module):
    def __init__(self, *, element_embedding_dim, num_element, num_classes, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        torch.set_default_tensor_type(torch.FloatTensor)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_element + 1, element_embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, element_embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)
        
   
        self.blocks = nn.ModuleList([
            Block(element_embedding_dim, heads, dim_head, mlp_dim, dropout) for i in range(depth)
        ])
        
       
        self.transformer = self.blocks
        
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head_ci = nn.Sequential(
            nn.LayerNorm(element_embedding_dim),
            nn.Linear(element_embedding_dim, element_embedding_dim),
            nn.Linear(element_embedding_dim, num_classes)
        )
        self.mlp_head_csvd = nn.Sequential(
            nn.LayerNorm(element_embedding_dim),
            nn.Linear(element_embedding_dim, num_classes)
        )
    def forward(self, x, pretrain=False, register_hook=False):
        if not pretrain:
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            for blk in self.blocks:
                x = blk(x, register_hook=register_hook)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            feature = x
        return self.mlp_head_csvd(x), self.mlp_head_ci(x), feature 