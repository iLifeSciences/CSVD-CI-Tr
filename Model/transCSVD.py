import torch
from torch import nn, einsum
from einops import rearrange, repeat
from Model.vit import ViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BatchNorm = nn.BatchNorm1d
num_in_ch = 3

class TransCSVD(nn.Module):
    def __init__(self, *, element_embedding_dim, num_element, image_size, patch_size, num_classes, depth, heads,
                 mlp_dim, img_dim=None, pool='cls', dim_head=32, dropout=0., emb_dropout=0.,
                 pre_train=False, load_weight=False, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.pre_train = pre_train
        self.ele_embedding_dim = element_embedding_dim
        self.ele_embedding = ElementPaddingBlock(element_embedding_dim, verbose=verbose)
        self.vit = ViT(element_embedding_dim=element_embedding_dim, num_element=num_element, num_classes=num_classes,
                       depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head, dropout=dropout,
                       emb_dropout=emb_dropout)

    def forward(self, radiomics=None, rad_names=None, register_hook=False):
        x = torch.zeros(0).to(device)
        if radiomics is not None:
            x_temp = self.ele_embedding(radiomics, rad_names)
            x = torch.cat((x, x_temp), dim=1)
        print('x.size():',x.size())
        out = self.vit(x, register_hook=register_hook)
        return out, x

class ElementEmbeddingLayer(nn.Module):
    def __init__(self, in_dim, element_embedding_dim, verbose=False):
        super().__init__()
        self.eel = nn.Linear(in_dim, element_embedding_dim)
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            print(x.size())
        return self.eel(x)

class ElementPaddingLayer(nn.Module):
    def __init__(self, in_dim, element_embedding_dim):
        super().__init__()
        if in_dim > element_embedding_dim:
            self.num_pad = in_dim // element_embedding_dim + 2
        else:
            self.num_pad = 1
        self.pad = int(element_embedding_dim - in_dim / self.num_pad)

    def forward(self, x):
        x = rearrange(x, 'b (c w) -> b c w', c=self.num_pad)
        return torch.cat((x, torch.zeros(x.size(0), x.size(1), self.pad).to(device)), dim=2)

class ElementConcatingBlock(nn.Module):
    def __init__(self, element_dim, element_out_dim, verbose=False):
        super(ElementConcatingBlock, self).__init__()
        self.element_dim = element_dim
        self.verbose = verbose
        self.embedding_layer = nn.Linear(element_dim, element_out_dim)

    def forward(self, data):
        out_data = torch.zeros(0).to(device)
        for i in range(len(data)):
            if self.verbose:
                print(data[i].shape)
            out_data = torch.cat((out_data, data[i].to(device)), dim=1)
        padding = self.element_dim - (out_data.size(1) % self.element_dim)
        if self.verbose:
            print('{} with padding {}'.format(out_data.size(), padding))
        out_data = torch.cat((out_data, torch.zeros((out_data.size(0), padding)).to(device)), dim=1)
        out_data = rearrange(out_data, 'b (b1 d) -> b b1 d', d=self.element_dim)
        out_data = self.embedding_layer(out_data)
        if self.verbose:
            print('->', out_data.size())
        if self.verbose:
            print('=' * 10)
        return out_data

class ElementPaddingBlock(nn.Module):
    def __init__(self, element_padding_dim, verbose=False):
        super(ElementPaddingBlock, self).__init__()
        self.element_padding_dim = element_padding_dim
        self.element_padding_5 = ElementPaddingLayer(5, element_padding_dim)
        self.element_padding_14 = ElementPaddingLayer(14, element_padding_dim)
        self.element_padding_16 = ElementPaddingLayer(16, element_padding_dim)
        self.element_padding_17 = ElementPaddingLayer(17, element_padding_dim)
        self.element_padding_18 = ElementPaddingLayer(18, element_padding_dim)
        self.element_padding_24 = ElementPaddingLayer(24, element_padding_dim)
        self.element_padding_68 = ElementPaddingLayer(68, element_padding_dim)
        self.verbose = verbose

    def forward(self, data, rad_names):
        out_data = torch.zeros(0).to(device)
        for i in range(len(data)):
            length = data[i].shape[1]
            if self.verbose:
                print(data[i].shape, end=' ')
            if length == 5:
                out_data = torch.cat((out_data, self.element_padding_5(data[i].to(device))), dim=1)
            elif length == 14:
                out_data = torch.cat((out_data, self.element_padding_14(data[i].to(device))), dim=1)
            elif length == 16:
                out_data = torch.cat((out_data, self.element_padding_16(data[i].to(device))), dim=1)
            elif length == 17:
                out_data = torch.cat((out_data, self.element_padding_17(data[i].to(device))), dim=1)
            elif length == 18:
                out_data = torch.cat((out_data, self.element_padding_18(data[i].to(device))), dim=1)
            elif length == 24:
                out_data = torch.cat((out_data, self.element_padding_24(data[i].to(device))), dim=1)
            elif length == 68:
                out_data = torch.cat((out_data, self.element_padding_68(data[i].to(device))), dim=1)
            else:
                print(rad_names[i], data[i].size())
                assert 0, 'The dimension is not expected! That should be in the list: [5, 14, 16, 17, 18, 24, 68]'
            if self.verbose:
                print('->', out_data[:, -1, ].size())
        if self.verbose:
            print('=' * 10)
        return out_data

class ElementEmbeddingBlock(nn.Module):
    def __init__(self, element_embedding_dim, verbose=False):
        super(ElementEmbeddingBlock, self).__init__()
        self.element_embedding_dim = element_embedding_dim
        self.element_embedding_14 = ElementEmbeddingLayer(14, element_embedding_dim)
        self.element_embedding_16 = ElementEmbeddingLayer(16, element_embedding_dim)
        self.element_embedding_18 = ElementEmbeddingLayer(18, element_embedding_dim)
        self.element_embedding_24 = ElementEmbeddingLayer(24, element_embedding_dim)
        self.element_embedding_68 = ElementEmbeddingLayer(68, element_embedding_dim)
        self.verbose = verbose

    def forward(self, data):
        out_data = torch.zeros((data[0].shape[0], len(data), self.element_embedding_dim)).to(device)
        for i in range(len(data)):
            length = data[i].shape[1]
            if self.verbose:
                print(data[i].shape, end=' ')
            if length == 14:
                out_data[:, i, ] = self.element_embedding_14(data[i].to(device))
            elif length == 16:
                out_data[:, i, ] = self.element_embedding_16(data[i].to(device))
            elif length == 18:
                out_data[:, i, ] = self.element_embedding_18(data[i].to(device))
            elif length == 24:
                out_data[:, i, ] = self.element_embedding_24(data[i].to(device))
            elif length == 68:
                out_data[:, i, ] = self.element_embedding_68(data[i].to(device))
            else:
                assert 0, 'The dimension is not expected! That should be in the list: [14, 16, 18, 24, 68]'
            if self.verbose:
                print('->', out_data[:, i, ].size())
        if self.verbose:
            print('=' * 10)
        return out_data 