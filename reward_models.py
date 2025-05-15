import random
from typing import Callable, List, Union

import autoroot
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_bleu import BLEU, SelfBLEU
from pykeops.torch import Vi, Vj
from transformers import AutoModel, AutoTokenizer

from utils.common import disable_grad, split_batch
from utils.net import MLP


pbe = lambda x, y, k: ((Vi(x) - Vj(y)) ** 2).sum().Kmin(k, 1)
tdiv_fn = lambda x, y: -(-(Vi(x) - Vj(y)).abs().sum() / 100).logsumexp(1).exp()
cos_sim = lambda x, y: (Vi(x) * Vj(y)).sum().sum_reduction(1)
cos_dist_min = lambda x, y, k: (-Vi(x) * Vj(y)).sum().Kmin(k, 1)


# from https://github.com/Improbable-AI/curiosity_redteam
class SelfBLEUReward:
    def __init__(
        self,
        grams: List[int] = [2, 3, 4, 5],  # noqa: B006
        sample_size: int = -1,
        tokenizer=None,
        device: Union[str, torch.device] = "cuda",
    ):
        self.references = []
        self.grams = grams
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.device = device

    def get_references(self, hypotheses):
        if self.tokenizer is None:
            return list(map(nltk.word_tokenize, hypotheses))
        else:
            return self.tokenizer(hypotheses)["input_ids"]

    def __call__(self, tokenized_hypotheses):
        weights = {f"{n}-gram": ([1.0 / n] * n) for n in self.grams}
        if self.sample_size > 0:
            sample_size = min(len(self.references), self.sample_size)
            bleu = BLEU(random.sample(self.references, k=sample_size), weights)
        else:
            bleu = BLEU(self.references, weights)
        scores = list(bleu.get_score(tokenized_hypotheses).values())
        return -torch.tensor(scores).mean(0).to(self.device)

    def get_selfbleu_score(self, sentences):
        tokenized_hypotheses = self.get_references(sentences)
        weights = {f"{n}-gram": ([1.0 / n] * n) for n in self.grams}
        scores = list(SelfBLEU(tokenized_hypotheses, weights).get_score().values())
        scores = torch.tensor(scores).mean(0).to(self.device)
        for i in range(len(tokenized_hypotheses)):
            if len(tokenized_hypotheses[i]) <= 5:
                scores[i] = 1
        return -scores


class SentenceEmbeddingReward:
    def __init__(self, device="cuda", k = 5):
        self.device = device
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.s_embeddings = None
        self.k = k

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def get_embeddings(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, return_tensors="pt", truncation=True).to(self.device)
        x = F.normalize(self.mean_pooling(self.model(**inputs), inputs["attention_mask"]), p=2, dim=1)
        return x

    def cosine_similarity(self, x=None, y=None):
        if y is None:
            assert self.s_embeddings is not None
            y = self.s_embeddings
        if x is None:
            x = y
        return cos_dist_min(x, y, self.k + 1)[:, 1:].mean(-1)

    def TDiv(self, x=None, y=None):
        if y is None:
            assert self.s_embeddings is not None
            y = self.s_embeddings
        if x is None:
            x = y
        return tdiv_fn(x, y).squeeze() / len(y)


# Particle-Based Estimator (PBE)
class PBE:
    def __init__(self, k=5, sample_size=-1):
        self.k = k
        self.sample_size = sample_size
        self.token_embeddings = []

    def __call__(self, x, use_buffer=False):
        B, L, D = x.shape
        x = x.reshape(-1, D)
        if not use_buffer or len(self.token_embeddings) == 0:
            y = x
        else:
            if self.sample_size > 0:
                sample_size = min(len(x), self.sample_size)
                token_embeddings = random.sample(self.token_embeddings, k=sample_size)
            else:
                token_embeddings = self.token_embeddings
            y = torch.stack(token_embeddings).reshape(-1, D)
        r = pbe(x, y, self.k + 1)[:, 1:].sqrt()
        rew = (1 + r.mean(-1)).log()
        return rew.reshape(B, L)


# Random Network Distillation (RND)
class RND(nn.Module):
    def __init__(self, obs_dim, mlp_hidden_dims, obs_rep_dim, lr=1e4, epoch=4):
        super().__init__()
        self.encoder = MLP([obs_dim, *mlp_hidden_dims, obs_rep_dim])
        self.target_encoder = MLP([obs_dim, *mlp_hidden_dims, obs_rep_dim])
        disable_grad(self.target_encoder)
        self.mse = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.epoch = epoch

    def forward(self, obs):
        shape = obs.shape
        obs = obs.reshape(-1, shape[-1])
        loss = self.mse(self.encoder(obs), self.target_encoder(obs)).mean(-1)
        return loss.reshape(*shape[:-1])

    def update(self, obs):
        for _ in range(self.epoch):
            for idx in split_batch(32, obs.shape[0]):
                self.optimizer.zero_grad()
                loss = self(obs[idx]).mean()
                loss.backward()
                self.optimizer.step()
