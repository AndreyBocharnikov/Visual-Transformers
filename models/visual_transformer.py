import typing as tp

import torch
import torch.nn as nn


class FilterBasedTokenizer(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int = 1024, n_visual_tokens: int = 8) -> None:
        super().__init__()
        self.feature_map_cs, self.visual_tokens_cs, self.n_visual_tokens = feature_map_cs, visual_tokens_cs, n_visual_tokens
        self.W_a = nn.Linear(feature_map_cs, n_visual_tokens)
        self.W_v = nn.Linear(feature_map_cs, visual_tokens_cs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X: torch.Tensor, T_in: tp.Optional[torch.Tensor]):
        # X.shape = bs, HW, feature_map_cs
        A = self.W_a(X)  # bs, HW, n_visual_tokens
        V = self.W_v(X)  # bs, HW, visual_tokens_cs
        attention_weights = self.softmax(A).permute(0, 2, 1)  # bs, n_visual_tokens, HW
        return torch.bmm(attention_weights, V)  # bs, n_visual_tokens, visual_tokens_cs


class RecurrentTokenizer(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int = 1024):
        super().__init__()
        self.feature_map_cs, self.visual_tokens_cs = feature_map_cs, visual_tokens_cs
        self.W_tr = nn.Linear(visual_tokens_cs, feature_map_cs)
        self.W_v = nn.Linear(feature_map_cs, visual_tokens_cs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X: torch.Tensor, T_in: torch.Tensor):
        # X.shape = bs, HW, feature_map_cs
        # T_in.shape = bs, n_visual_tokens, visual_tokens_cs
        W_r = self.W_tr(T_in).permute(0, 2, 1)  # bs, feature_map_cs, n_visual_tokens
        proj = torch.bmm(X, W_r)  # bs, HW, n_visual_tokens
        attention_weights = self.softmax(proj).permute(0, 2, 1)  # bs, n_visual_tokens, HW
        values = self.W_v(X)
        return torch.bmm(attention_weights, values)


class Transformer(nn.Module):
    def __init__(self, visual_tokens_cs: int):
        super().__init__()
        self.key_proj = nn.Linear(visual_tokens_cs, visual_tokens_cs // 2)
        self.query_proj = nn.Linear(visual_tokens_cs, visual_tokens_cs // 2)
        self.value_proj = nn.Linear(visual_tokens_cs, visual_tokens_cs)
        self.f1 = nn.Linear(visual_tokens_cs, visual_tokens_cs)
        self.f2 = nn.Linear(visual_tokens_cs, visual_tokens_cs)
        self.softmax = nn.Softmax(dim=2)
        self.activation = nn.ReLU()

    def forward(self, T_in: torch.Tensor):
        key_query_product = torch.bmm(self.key_proj(T_in), self.query_proj(T_in).permute(0, 2, 1))  # bs, n_visual_tokens, n_visual_tokens
        values = self.value_proj(T_in)  # bs, n_visual_tokens, feature_map_cs
        T_out_dash = T_in + torch.bmm(self.softmax(key_query_product), values)
        return T_out_dash + self.f2(self.activation(self.f1(T_out_dash)))


class Projector(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int):
        super().__init__()
        self.W_q = nn.Linear(feature_map_cs, feature_map_cs)
        self.W_k = nn.Linear(visual_tokens_cs, feature_map_cs)
        self.W_v = nn.Linear(visual_tokens_cs, feature_map_cs)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X_in: torch.Tensor, T: torch.Tensor):
        pixel_token_similarity = torch.bmm(self.W_q(X_in), self.W_k(T).permute(0, 2, 1))
        values = self.W_v(T)
        attention_weights = self.softmax(pixel_token_similarity)
        return X_in + torch.bmm(attention_weights, values)


class VisualTransformer(nn.Module):
    def __init__(self, tokenizer: tp.Union[FilterBasedTokenizer, RecurrentTokenizer], is_last):
        super().__init__()
        self.is_last = is_last
        self.tokenizer = tokenizer
        self.transformer = Transformer(tokenizer.visual_tokens_cs)
        if not is_last:
            self.projector = Projector(tokenizer.feature_map_cs, tokenizer.visual_tokens_cs)

    def forward(self, feature_map: torch.Tensor, T_in: tp.Optional[torch.Tensor]):
        visual_tokens = self.tokenizer(feature_map, T_in)
        visual_tokens = self.transformer(visual_tokens)
        if self.is_last:
            return visual_tokens
        else:
            return self.projector(feature_map, visual_tokens), visual_tokens
