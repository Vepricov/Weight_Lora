from typing import List

import numpy as np
import torch


class MarkovStateBase:
    """
    Base for Markov probability correction
    Uniform distribution
    """

    def __init__(self, d: int, device="cuda", **params) -> None:
        self.params = params
        self.d = d
        self.default_prob = torch.ones(d, device=device) / d
        self.send_history = []
        self.history_max_size = 1

    def get_proba(self, x) -> torch.Tensor:
        return self.default_prob

    def update_state(self, last_selection_mask: torch.Tensor) -> None:
        if len(self.send_history) >= self.history_max_size:
            self.send_history = self.send_history[-self.history_max_size :]
        self.send_history.append(last_selection_mask)
        # debug check for banlastk=1
        # if len(self.send_history) > 1:
        #     assert np.max(self.send_history[-1] + self.send_history[-2]) < 2


class MarkovStateBanLastK(MarkovStateBase):
    """
    Markov state that bans coordinates from last K steps
    from being transferred

    XXX: can be used only with compressors which rate is:
    rate * (K + 1) <= 1
    """

    def __init__(self, d, k=0, **params) -> None:
        super().__init__(d=d, **params)
        self.ban_last_k = k
        self.history_max_size = k + 1

    # probability of i-th to be compressed
    # for those we DID NOT sent in send_history, prob of being compressed should be 0
    def get_proba(self, x: torch.Tensor = None):
        """
        x input shape: (batch_size, flattened_hidden_size)
        """
        prob = self.default_prob.clone()
        if self.ban_last_k == 0:
            return prob
        # set all last K iterations to 0, distribute over others
        ban_mask = torch.zeros_like(prob, dtype=bool)
        for element in self.send_history[-self.ban_last_k :]:
            ban_mask = torch.logical_or(ban_mask, element)
        num_banned = ban_mask.sum()
        banned_prob_sum = prob[ban_mask].sum()
        prob[ban_mask] = 0
        prob[~ban_mask] += banned_prob_sum / (self.d - num_banned)
        assert torch.isclose(prob.sum(), torch.ones(1, device=prob.device))
        return prob


def projection_simplex(x):
    x_sort, _ = torch.sort(x, descending=True)
    summa = torch.cumsum(x_sort, dim=0)
    indices = torch.arange(1, len(x_sort) + 1, dtype=torch.float64, device=x.device)
    comp = x_sort + 1 / indices * (1 - summa)

    rho = torch.where(comp > 0)[0][-1] if torch.any(comp > 0) else 0
    summa_ans = summa[rho]

    lamb = 1 / (rho + 1) * (1 - summa_ans)

    x_next = torch.clamp(x + lamb, min=0)

    return x_next


class MarkovStateKawasaki(MarkovStateBase):
    """
    Markov state that decreases probability of entries that
    were recently transferred
    """

    def __init__(
        self, d, func="softmax", k=1, temp=0.01, base_of_power=2, **params
    ) -> None:
        super().__init__(d=d, **params)
        functions = {
            "softmax": lambda x, t=temp: torch.exp(x / t) / torch.sum(torch.exp(x / t)),
            "devide_sum": lambda x: x / x.sum(),
            "proj": lambda x: projection_simplex(x),
        }

        self.to_simplex = functions[func]
        self.last_k = k
        self.base_of_power = base_of_power
        self.history_max_size = k + 1

    # probability of i-th to be compressed
    # for those we DID NOT sent in send_history, prob of being compressed should be 0
    def get_proba(self, x: torch.Tensor) -> None:
        """
        x input shape: (batch_size, flattened_hidden_size)
        """
        prob = self.default_prob.clone()
        if self.last_k == 0:
            return prob
        # set all last K iterations to 0, distribute over others
        communicated_times = torch.zeros_like(prob, dtype=torch.int)
        for element in self.send_history[-self.last_k :]:
            communicated_times += element
        prob = prob / (self.base_of_power**communicated_times)

        prob_normalized = self.to_simplex(prob)
        assert torch.isclose(prob_normalized.sum(), torch.ones(1, device=prob.device))
        return prob_normalized


class Compressor:
    name: str = None  # compressor name
    compression_rate: float = 1.0  # ratio of information sent, should be in [0;1]
    idx: torch.Tensor = (
        None  # indicies of compressed communicated during last iteration
    )

    def __init__(
        self, name="identity", compression_rate=1.0, **optional_params
    ) -> None:
        self.name = name
        self.compression_rate = compression_rate
        self.idx = None  # mask of indices that we compressed last time (set to 0 in randk/topk)

    def get_last_selection_mask(self) -> torch.Tensor:
        return self.idx

    def compress(self, x: np.ndarray) -> torch.Tensor:
        raise NotImplementedError()  # should not be called for base class


class Identity(Compressor):
    """
    Identity compressor
    """

    def __init__(self, **compress_params) -> None:
        super().__init__(**compress_params)

    def compress(self, x: torch.Tensor, prob: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Return input vector `x`. Input probability prior `prob` is ignored.

        x shape: (batch_size, flattened_hidden_size)
        prob shape: (flattened_hidden_size)
        output shape: (batch_size, flattened_hidden_size)
        """
        # set idx for last selection mask
        self.idx = torch.ones(x.shape[0], device=x.device)
        return x


class Randk(Compressor):
    """
    RandK unbiased compressor with probability prior, applied to batched input.
    """

    def __init__(self, **compress_params) -> None:
        super().__init__(**compress_params)

    def compress(self, x: torch.Tensor, prob: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compress input (batch) vector `x`, given probability prior `prob` for each dimension.
        Randomly selects indices for communication based on `prob`, and then filters all other
        indices in each vector from batch to zero.

        x shape: (batch_size, flattened_hidden_size)
        prob shape: (flattened_hidden_size)
        output shape: (batch_size, flattened_hidden_size)
        """
        d = x.shape[0]
        n_out = int(self.compression_rate * d)
        idx_selected = prob.multinomial(num_samples=n_out, replacement=False)
        self.idx = torch.zeros(d, dtype=torch.bool, device=x.device)
        self.idx[idx_selected] = 1
        x[~self.idx] = 0
        return x / self.compression_rate

class NaturalCompressor(Compressor):
    """
    Natural probabilistic compression implementation.
    XXX: simply cuts out mantissa to 3rd sign (but not sign). Does not depend on compression rate.
    """

    def __init__(self, mantissa_cutoff: int = 7, **compress_params) -> None:
        super().__init__(**compress_params)
        self.mantissa_cutoff = mantissa_cutoff

    def compress(self, x: torch.Tensor, prob: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Round mantissa in input Tensor `x`. Does not rely on probabilities.
        """
        mantissa, exponent = torch.frexp(x)
        mantissa = torch.round(mantissa, decimals=self.mantissa_cutoff)
        return torch.ldexp(mantissa, exponent).to(device=x.device, dtype=x.dtype)


class PermutationCompressor(Compressor):
    """
    Given probabilites of choosing each coordinates and mask of those coordinates that
    are banned by permutation rule, compress only those which are available
    """

    def __init__(self, **compress_params) -> None:
        super().__init__(**compress_params)
    
    """
    mask - 0 or 1 for each value in the array
    """
    def compress(self, x: torch.Tensor, prob: torch.Tensor, used_idx_mask: torch.Tensor) -> torch.Tensor:
        assert prob is not None and used_idx_mask is not None
        d = x.shape[0]
        n_out = int(self.compression_rate * d)

        prob_cur = ((~used_idx_mask).to(dtype=torch.float64)) / (d - used_idx_mask.sum())

        idx_selected = prob_cur.multinomial(num_samples=n_out, replacement=False)
        self.idx = torch.zeros(d, dtype=torch.bool, device=x.device)
        self.idx[idx_selected] = 1
        x[~self.idx] = 0
        return x / self.compression_rate

class MarkovCompressor:
    """
    Compressor with Markov rules on selecting coordinates
    Incorporates two components:
        - MarkovState for handling history and setting priors
        - Compressor to perform prior-based compression
    """

    compressor: Compressor
    markov_state: MarkovStateBase

    def __init__(
        self, compressor_params: dict, markov_params: dict, d: int, device="cpu"
    ) -> None:
        compressor_name = compressor_params["name"]
        compressor_dict = {
            "identity": Identity,
            "randk": Randk,
            "permutation": PermutationCompressor,
            "natural": NaturalCompressor,
        }
        markov_name = markov_params["name"]
        markov_dict = {
            "identity": MarkovStateBase,
            "banlastk": MarkovStateBanLastK,
            "kawasaki": MarkovStateKawasaki,
        }
        self.compressor = compressor_dict[compressor_name](
            device=device, **compressor_params
        )
        self.markov_state = markov_dict[markov_name](d, device=device, **markov_params)

    def compress_(self, x: torch.Tensor, prob=None, used_idx_mask=None) -> torch.Tensor:
        return self.compressor.compress(x, prob=prob, used_idx_mask=used_idx_mask)

    # compress single batch vector
    def compress(self, x: torch.Tensor, training=True, used_idx_mask=None, **kwargs):
        """
        Compresses input `x` tensor in two stages: selecting coordinate choice prior probability
        with MarkovState, and then selecting coordinates at random with Compressor.
        In simulated scenario, compressed coordinates in each batch are set to 0.

        x shape: (batch_size, flattened_hidden_size)
        output shape: (batch_size, flattened_hidden_size)

        """
        ### XXX: hack for small bias terms
        # print(x.numel())
        # if x.numel() < 100:
        #     return x
        x_flattened = x.reshape(-1)
        if training:
            prob = self.markov_state.get_proba(x_flattened)
        else:
            prob = self.markov_state.default_prob

        x_compressed = self.compress_(x_flattened, prob=prob, used_idx_mask=used_idx_mask)

        if training:
            selection_mask = self.compressor.get_last_selection_mask()
            self.markov_state.update_state(selection_mask)

        return x_compressed.reshape(x.shape)