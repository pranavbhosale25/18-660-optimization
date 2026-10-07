"""Provided environment: simultaneous one-round, two-card poker.

All cards are dealt uniformly without replacement. The community card is
public BEFORE simultaneous bids. A player knows their own hole card only.
Both ante one chip; bids are extra commitments. Unmatched excess is returned.
"""
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PokerRules:
    ranks: int = 3
    suits: int = 2
    ante: float = 1.0
    bids: tuple = (0, 1, 2)

    def __post_init__(self):
        if self.ranks < 2 or self.suits < 2 or self.ranks*self.suits < 3:
            raise ValueError("Use at least two ranks, two suits, and three cards")
        if (not isinstance(self.bids, tuple) or len(self.bids) < 2 or
                not np.isfinite(self.bids).all() or min(self.bids) < 0 or
                any(a >= b for a, b in zip(self.bids, self.bids[1:]))):
            raise ValueError("bids must be an increasing tuple of nonnegative finite amounts")
        if not np.isfinite(self.ante) or self.ante <= 0:
            raise ValueError("A positive ante avoids a free-check degeneracy")

    @property
    def deck_size(self):
        return self.ranks * self.suits


class PokerData(NamedTuple):
    kernels: jax.Array  # (community, hole_type*action, hole_type*action)
    holes: jax.Array    # (community, hole_type), excludes community card


def hand_score(hole, community, ranks: int, suits: int):
    """Lexicographic category/high/low encoding: pair > flush > high card.

    Rank 0 is lowest. Suits never break ties. All arguments may be arrays.
    """
    hr, cr = hole // suits, community // suits
    pair = hr == cr
    flush = (hole % suits) == (community % suits)
    category = jnp.where(pair, 2, jnp.where(flush, 1, 0))
    high, low = jnp.maximum(hr, cr), jnp.minimum(hr, cr)
    return category * ranks**2 + high * ranks + low


def payoff(community, hole1, hole2, action1, action2, rules=PokerRules()):
    """Net transfer TO player 1 (player 2 receives its negative)."""
    s1 = hand_score(hole1, community, rules.ranks, rules.suits)
    s2 = hand_score(hole2, community, rules.ranks, rules.suits)
    showdown = jnp.sign(s1-s2)
    winner = jnp.where(action1 == action2, showdown, jnp.sign(action1-action2))
    return winner * (rules.ante + jnp.minimum(action1, action2))


def build_poker(rules=PokerRules()) -> PokerData:
    """Vectorize all public boards, private deals, and action pairs exactly.

    No Monte Carlo and no exponential enumeration of pure contingent plans.
    The six-card default has 6 boards, 5 types/player and 15x15 kernels.
    A standard deck has 52 boards and 153x153 kernels.
    """
    D = rules.deck_size
    cards = np.arange(D)
    B = len(rules.bids)
    bids = jnp.asarray(rules.bids)
    holes = jnp.asarray(np.stack([cards[cards != c] for c in cards]))
    def board(c, h):
        # Axes i,a,j,b match flattening of conditional strategies.
        hi, hj = h[:, None, None, None], h[None, None, :, None]
        a, b = bids[None, :, None, None], bids[None, None, None, :]
        rewards = payoff(c, hi, hj, a, b, rules)
        joint = (hi != hj).astype(jnp.float64) / ((D-1)*(D-2))
        return (rewards * joint).reshape((B*(D-1), B*(D-1)))
    kernels = jax.jit(jax.vmap(board))(jnp.arange(D), holes)
    return PokerData(kernels, holes)


def exact_payoff(data: PokerData, row, col):
    """row/col: (community, hole_type, number_of_bids). Average over public boards."""
    r, c = row.reshape(len(row), -1), col.reshape(len(col), -1)
    return jnp.einsum("bi,bij,bj->", r, data.kernels, c) / len(row)


def policy_full_deck(data, policies):
    """Convert compact policy to [public_card, private_card, action].

    Impossible public==private entries are zero, never queried by the dealer.
    """
    D = data.holes.shape[0]
    out = jnp.zeros((D, D, policies.shape[-1]), dtype=policies.dtype)
    return out.at[jnp.arange(D)[:, None], data.holes].set(policies)


def sample_payoffs(key, data, row, col, samples: int, rules=PokerRules()):
    """Vectorized independent deals; for demonstration, not equilibrium grading."""
    keys = jax.random.split(key, 5)
    D = rules.deck_size
    board = jax.random.randint(keys[0], (samples,), 0, D)
    i = jax.random.randint(keys[1], (samples,), 0, D-1)
    # Uniformly choose opponent type except our private card.
    j0 = jax.random.randint(keys[2], (samples,), 0, D-2)
    j = j0 + (j0 >= i)
    a = jax.random.categorical(keys[3], jnp.log(row[board, i]), axis=-1)
    b = jax.random.categorical(keys[4], jnp.log(col[board, j]), axis=-1)
    return payoff(board, data.holes[board, i], data.holes[board, j],
                  jnp.asarray(rules.bids)[a], jnp.asarray(rules.bids)[b], rules)
