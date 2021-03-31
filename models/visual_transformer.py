import typing as tp

import torch
import torch.nn as nn


class FilterBasedTokenizer(nn.Module):
    def __init__(self, in_channels: int, n_visual_tokens: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.pointwise_conv = nn.Linear(in_channels, n_visual_tokens)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X: torch.Tensor, T_in: tp.Optional[torch.Tensor]):
        # X.shape = bs, HW, in_channels
        proj = self.pointwise_conv(X)  # bs, HW, n_visual_tokens
        attention_weights = self.softmax(proj).permute(0, 2, 1)  # bs, n_visual_tokens, HW
        return torch.bmm(attention_weights, X)  # bs, n_visual_tokens, in_channels


class RecurrentTokenizer(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.W_tr = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X: torch.Tensor, T_in: torch.Tensor):
        # X.shape = bs, HW, in_channels
        # T_in.shape = bs, n_visual_tokens, in_channels
        W_r = self.W_tr(T_in).permute(0, 2, 1)  # bs, in_channels, n_visual_tokens
        proj = torch.bmm(X, W_r)  # bs, HW, n_visual_tokens
        attention_weights = self.softmax(proj).permute(0, 2, 1)  # bs, n_visual_tokens, HW
        return torch.bmm(attention_weights, X)


class Transformer(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.key_proj = nn.Linear(in_channels, in_channels // 2)
        self.query_proj = nn.Linear(in_channels, in_channels // 2)
        self.f1 = nn.Linear(in_channels, in_channels)
        self.f2 = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=2)
        self.activation = nn.ReLU()

    def forward(self, T_in: torch.Tensor):
        key_query_product = torch.bmm(self.key_proj(T_in), self.query_proj(T_in).permute(0, 2, 1))  # bs, n_visual_tokens, n_visual_tokens
        T_out_dash = T_in + torch.bmm(self.softmax(key_query_product), T_in)
        return T_out_dash + self.f2(self.activation(self.f1(T_out_dash)))


class Projector(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X_in: torch.Tensor, T: torch.Tensor):
        pixel_token_similarity = torch.bmm(self.W_q(X_in), self.W_k(T).permute(0, 2, 1))
        attention_weights = self.softmax(pixel_token_similarity)
        return X_in + torch.bmm(attention_weights, T)


class VisualTransformer(nn.Module):
    def __init__(self, tokenizer: tp.Union[FilterBasedTokenizer, RecurrentTokenizer], is_last):
        super().__init__()
        self.is_last = is_last
        self.tokenizer = tokenizer
        self.transformer = Transformer(tokenizer.in_channels)
        if not is_last:
            self.projector = Projector(tokenizer.in_channels)

    def forward(self, feature_map: torch.Tensor, T_in: tp.Optional[torch.Tensor]):
        #print("VT input", feature_map.shape)
        visual_tokens = self.tokenizer(feature_map, T_in)
        #print("tokenizer output", visual_tokens.shape)
        visual_tokens = self.transformer(visual_tokens)
        if self.is_last:
            return visual_tokens
        else:
            return self.projector(feature_map, visual_tokens), visual_tokens
