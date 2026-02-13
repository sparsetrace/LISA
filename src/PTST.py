# PTST.py
# ============================================================
# PTST: Patch Time-Series Transformer (PatchTST-style) in JAX/Flax
#
# Design goals:
# - Same API style as your LISA/ALSA/TART:
#     model = PTST(R_tX_train, L=..., pred_len=..., ...)
#     preds = model(prefix, steps=H)
#
# - Training is automatic in __init__ (if R_tX is provided).
# - Inference is in __call__.
#
# Core ideas:
# - Patch along time: tokens are length-P patches, so seq_len reduces from L to L/P.
# - Channel-independence (default): train one univariate model shared across all channels.
#
# Forecasting:
# - Direct multi-horizon head for pred_len_train steps.
# - For steps > pred_len_train: rollout in pred_len_train-sized chunks (AR over chunks).
#
# Dependencies:
# - jax, flax, optax, numpy
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state


# ============================================================
# Utilities
# ============================================================

def zscore(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + eps
    return (x - mu) / sd, (mu, sd)


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    return X


def choose_heads(d_model: int, target_head_dim: int = 32, max_heads: int = 8) -> int:
    """Pick a reasonable number of heads given d_model."""
    n_heads = max(1, min(max_heads, d_model // target_head_dim))
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1
    return int(max(1, n_heads))

def make_windows_multihorizon(X: np.ndarray, L: int, H: int):
    """
    Build multihorizon dataset:
      X_in[i]  = X[i : i+L]        (L, D)
      Y_out[i] = X[i+L : i+L+H]    (H, D)

    Returns:
      X_in : (N, L, D)
      Y_out: (N, H, D)
    """
    X = _as_2d(X)
    T, D = X.shape
    if T < L + H + 1:
        raise ValueError(f"Need T >= L+H+1. Got T={T}, L={L}, H={H}.")
    N = T - (L + H) + 1

    X_in = np.stack([X[i : i + L] for i in range(N)], axis=0)          # (N,L,D)
    Y_out = np.stack([X[i + L : i + L + H] for i in range(N)], axis=0) # (N,H,D)
    return np.ascontiguousarray(X_in), np.ascontiguousarray(Y_out)

# ============================================================
# Flax modules: Encoder-style attention
# ============================================================

class MultiHeadSelfAttentionX(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        """
        x: (B, T, d_model)
        output: (B, T, d_model)
        """
        B, T, Dm = x.shape
        assert Dm == self.d_model
        assert self.d_model % self.n_heads == 0
        head_dim = self.d_model // self.n_heads

        qkv = nn.Dense(3 * self.d_model, use_bias=False, name="qkv")(x)  # (B,T,3*Dm)
        qkv = qkv.reshape(B, T, 3, self.n_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(head_dim)
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B,H,T,T)
        attn = nn.softmax(attn_logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (B,H,T,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        out = nn.Dense(self.d_model, name="out_proj")(out)
        out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
        return out

class MultiHeadSelfAttentionXX(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0

    # NEW: context-derived channel attention (your C^-)
    use_ctx_cam: bool = False
    cam_tau: float = 1.0
    cam_init_scale: float = 0.0  # gate starts at 0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        """
        x: (B, T, d_model)
        """
        B, T, Dm = x.shape
        assert Dm == self.d_model
        assert self.d_model % self.n_heads == 0
        hd = self.d_model // self.n_heads

        qkv = nn.Dense(3 * self.d_model, use_bias=False, name="qkv")(x)  # (B,T,3*Dm)
        qkv = qkv.reshape(B, T, 3, self.n_heads, hd).transpose(2, 0, 3, 1, 4)  # (3,B,H,T,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,T,hd)

        scale = 1.0 / math.sqrt(hd)
        attn_logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale  # (B,H,T,T)
        A_plus = nn.softmax(attn_logits, axis=-1)                  # row-normalized over j
        A_plus = nn.Dropout(rate=self.dropout)(A_plus, deterministic=deterministic)

        # R'_{ihX} = A^+_{hij} R_{jhX}
        ctx = jnp.einsum("bhij,bhjd->bhid", A_plus, v)             # (B,H,T,hd)

        # unfold (hX)->x: R'_{ix}
        out = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)  # (B,T,x=d_model)

        # NEW: build C^- from R' and apply R'' = R' C^-
        if self.use_ctx_cam:
            # S_{xx'} = sum_i R'_{ix} R'_{ix'}
            S = jnp.einsum("btx,bty->bxy", out, out) / jnp.maximum(T, 1)  # (B,x,x)

            # column-wise softmax (columns sum to 1): softmax over row axis
            C_minus = nn.softmax(S / self.cam_tau, axis=-2)               # (B,x,x)

            mixed = jnp.einsum("btx,bxy->bty", out, C_minus)              # (B,T,x)

            # gated residual so it starts as identity when init_scale=0
            alpha = self.param(
                "ctx_cam_scale",
                lambda k, s: jnp.array(self.cam_init_scale, jnp.float32),
                (),
            )
            out = out + alpha * (mixed - out)

        out = nn.Dense(self.d_model, name="out_proj")(out)
        out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
        return out

class MultiHeadSelfAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0

    # NEW: context-derived channel attention (your C^-)
    use_ctx_cam: bool = False
    cam_tau: float = 1.0
    cam_init_scale: float = 0.0  # gate starts at 0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        """
        x: (B, T, d_model)
        """
        B, T, Dm = x.shape
        assert Dm == self.d_model
        assert self.d_model % self.n_heads == 0
        hd = self.d_model // self.n_heads

        qkv = nn.Dense(3 * self.d_model, use_bias=False, name="qkv")(x)  # (B,T,3*Dm)
        qkv = qkv.reshape(B, T, 3, self.n_heads, hd).transpose(2, 0, 3, 1, 4)  # (3,B,H,T,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,T,hd)

        scale = 1.0 / math.sqrt(hd)
        attn_logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale  # (B,H,T,T)
        A_plus = nn.softmax(attn_logits, axis=-1)                  # row-normalized over j
        A_plus = nn.Dropout(rate=self.dropout)(A_plus, deterministic=deterministic)

        # R'_{ihX} = A^+_{hij} R_{jhX}
        ctx = jnp.einsum("bhij,bhjd->bhid", A_plus, v)             # (B,H,T,hd)

        # unfold (hX)->x: R'_{ix}
        out = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)  # (B,T,x=d_model)

        # NEW: build C^- from R' and apply R'' = R' C^-
        if self.use_ctx_cam:
            # S_{xx'} = sum_i R'_{ix} R'_{ix'}
            S  = jnp.einsum("btx,bty->bxy", out, out) / jnp.maximum(T, 1)  # (B,x,x)
            S2 = jnp.einsum("bii->bi", S)
            S  = S2[:,None,:] + S2[:,:,None] - 2*S
            S  = -1.*S

            # column-wise softmax (columns sum to 1): softmax over row axis
            C_minus = nn.softmax(S / self.cam_tau, axis=-2)               # (B,x,x)

            #mixed = jnp.einsum("btx,bxy->bty", out, C_minus)              # (B,T,x)
            # gated residual so it starts as identity when init_scale=0
            #alpha = self.param( "ctx_cam_scale", lambda k, s: jnp.array(self.cam_init_scale, jnp.float32),(),)
            #out = out + alpha * (mixed - out)
            out = jnp.einsum("btx,bxy->bty", out, C_minus)              # (B,T,x)

        out = nn.Dense(self.d_model, name="out_proj")(out)
        out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
        return out

class ChannelSelfAttention(nn.Module):
    d_model: int
    tau: float = 1.0
    col_norm: bool = True
    init_scale: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        B, T, D = x.shape

        S = jnp.einsum("btd,bte->bde", x, x) / jnp.maximum(T, 1)
        C_row = nn.softmax(S / self.tau, axis=-1)
        C = jnp.swapaxes(C_row, -1, -2) if self.col_norm else C_row

        scale = self.param(
            "scale",
            lambda k, s: jnp.array(self.init_scale, jnp.float32),
            (),
        )

        # Return a delta (identity at scale=0)
        return scale * ((x @ C) - x)

class FeedForward(nn.Module):
    d_model: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        hidden = int(self.d_model * self.mlp_ratio)
        x = nn.Dense(hidden)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        return x

class TransformerEncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0

    # toggles
    use_ctx_cam: bool = False
    use_ffn: bool = True

    # ctx-cam params
    cam_tau: float = 1.0
    cam_init_scale: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        # --- Token self-attention (+ optional ctx channel mix inside it) ---
        h = nn.LayerNorm(name="attn_ln")(x)
        h = MultiHeadSelfAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            use_ctx_cam=self.use_ctx_cam,
            cam_tau=self.cam_tau,
            cam_init_scale=self.cam_init_scale,
            name="mhsa",
        )(h, deterministic=deterministic)
        x = x + h

        # --- FFN (optional) ---
        if self.use_ffn:
            h2 = nn.LayerNorm(name="ffn_ln")(x)
            h2 = FeedForward(
                d_model=self.d_model, mlp_ratio=self.mlp_ratio, dropout=self.dropout
            )(h2, deterministic=deterministic)
            x = x + h2

        return x

class PTSTBackbone(nn.Module):
    """
    PatchTST-style backbone for univariate series (channel-independent training).

    Input:
      x: (B, L_eff, 1)

    Output:
      yhat: (B, pred_len)

    Options:
      - use_cam: enable context-derived channel mixing C^- inside MHSA (default False)
      - use_ffn: enable FeedForward inside each encoder block (default True)
    """
    L_eff: int
    patch_len: int
    d_model: int
    depth: int
    n_heads: int
    pred_len: int
    dropout: float = 0.0
    use_learned_pos: bool = True

    # block options
    use_cam: bool = False     # now means "use ctx-CAM inside MHSA"
    use_ffn: bool = True

    # ctx-CAM hyperparams (only used if use_cam=True)
    cam_tau: float = 1.0
    cam_init_scale: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        B, L, C = x.shape
        assert C == 1
        assert L == self.L_eff
        assert self.L_eff % self.patch_len == 0

        n_patches = self.L_eff // self.patch_len

        # patchify: (B, n_patches, patch_len)
        patches = x.reshape(B, n_patches, self.patch_len)

        # patch embed (patch_len -> d_model)
        h = nn.Dense(self.d_model, name="patch_embed")(patches)  # (B, n_patches, d_model)

        # learned pos emb over patches
        if self.use_learned_pos:
            pos = self.param(
                "pos_emb",
                nn.initializers.normal(stddev=0.02),
                (n_patches, self.d_model),
            )  # (n_patches, d_model)
            h = h + pos[None, :, :]

        # encoder blocks (MHSA with optional ctx-CAM inside it; optional FFN)
        for i in range(self.depth):
            h = TransformerEncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout,
                mlp_ratio=4.0,

                use_ctx_cam=self.use_cam,
                use_ffn=self.use_ffn,
                cam_tau=self.cam_tau,
                cam_init_scale=self.cam_init_scale,

                name=f"enc_block_{i}",
            )(h, deterministic=deterministic)

        h = nn.LayerNorm(name="final_ln")(h)

        # PatchTST-style flatten head
        h_flat = h.reshape(B, -1)  # (B, n_patches*d_model)
        yhat = nn.Dense(self.pred_len, name="head")(h_flat)  # (B, pred_len)
        return yhat


# ============================================================
# PTST: user-facing class
# ============================================================

@dataclass
class PTSTConfig:
    # data / windowing
    L: int = 128
    patch_len: int = 16
    pred_len: int = 96  # training horizon per forward pass

    # model
    d_model: int = 128
    depth: int = 4
    n_heads: Optional[int] = None
    dropout: float = 0.0
    use_learned_pos: bool = True

    # ablations
    use_ffn: bool = True          # keep/remove FFN
    use_cam: bool = False         # enable ctx-CAM inside MHSA (your C^-)

    # ctx-CAM hyperparams (only used if use_cam=True)
    cam_tau: float = 1.0
    cam_init_scale: float = 0.0   # gate; set 0.0 to start as identity

    # training
    seed: int = 0
    val_split: float = 0.1
    batch_size: int = 128
    max_epochs: int = 50
    init_lr: float = 3e-4
    min_lr: float = 1e-5
    lr_decay: float = 0.5
    tol_rel_improve: float = 1e-3
    patience: int = 5

class PTST:
    """
    PTST = PatchTST-style forecaster with TART-like API.

    Usage:
      model = PTST(F_train, L=160, patch_len=16, pred_len=120, d_model=128, depth=4)
      preds = model(prefix, steps=120)

    Notes:
    - Default is channel-independent: one univariate backbone shared across all dims.
    - Trains multi-horizon (pred_len) directly.
    - For steps > pred_len, it rolls forward in pred_len chunks.
    """

    def __init__(self, R_tX: np.ndarray, **kwargs):
        cfg = PTSTConfig(**kwargs)
        self.cfg = cfg

        R_tX = _as_2d(R_tX)
        T, D = R_tX.shape
        self.D = int(D)

        # enforce patch compatibility
        L = int(cfg.L)
        P = int(cfg.patch_len)
        if L < P:
            raise ValueError(f"L={L} must be >= patch_len={P}")
        L_eff = (L // P) * P
        if L_eff != L:
            # keep behavior explicit for reproducibility:
            # we just trim L down to nearest multiple of patch_len
            print(f"[PTST] warning: L={L} not divisible by patch_len={P}. Using L_eff={L_eff}.")
        self.L = int(L)          # requested
        self.L_eff = int(L_eff)  # actually used
        self.patch_len = int(P)

        pred_len = int(cfg.pred_len)
        if T <= self.L_eff + pred_len + 1:
            raise ValueError(f"Need T > L_eff + pred_len. Got T={T}, L_eff={self.L_eff}, pred_len={pred_len}.")

        # normalize train series channel-wise
        F_norm, (mu, sd) = zscore(R_tX)
        self._mu = mu
        self._sd = sd

        # build multihorizon dataset
        X_in, Y_out = make_windows_multihorizon(F_norm, L=self.L_eff, H=pred_len)  # (N,L,D), (N,H,D)
        N = X_in.shape[0]

        # channel-independent reshape: treat each channel as separate training sample
        # X_ci: (N*D, L_eff, 1)
        # Y_ci: (N*D, pred_len)
        # X_in: (N, L_eff, D), Y_out: (N, pred_len, D)
        if X_in.shape[1] != self.L_eff:
            raise ValueError(
                f"[PTST] X_in has wrong time length {X_in.shape[1]} (expected {self.L_eff}). "
                f"This usually means window construction is wrong."
            )
        
        # channel-independent: treat each channel as an independent training sample
        # X_ci: (N*D, L_eff, 1)
        # Y_ci: (N*D, pred_len)
        X_ci = np.transpose(X_in, (0, 2, 1))              # (N, D, L_eff)
        X_ci = X_ci.reshape(N * D, self.L_eff)[:, :, None] # (N*D, L_eff, 1)
        
        Y_ci = np.transpose(Y_out, (0, 2, 1))             # (N, D, pred_len)
        Y_ci = Y_ci.reshape(N * D, pred_len)              # (N*D, pred_len)


        # train/val split
        n_val = max(1, int(cfg.val_split * (N * D)))
        n_tr = (N * D) - n_val
        Xtr, Ytr = X_ci[:n_tr], Y_ci[:n_tr]
        Xva, Yva = X_ci[n_tr:], Y_ci[n_tr:]

        # build model
        d_model = int(max(32, cfg.d_model))
        n_heads = cfg.n_heads if cfg.n_heads is not None else choose_heads(d_model)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.depth = int(cfg.depth)
        self.pred_len = int(pred_len)

        self.model = PTSTBackbone(
            L_eff=self.L_eff,
            patch_len=self.patch_len,
            d_model=self.d_model,
            depth=self.depth,
            n_heads=self.n_heads,
            pred_len=self.pred_len,
            dropout=float(cfg.dropout),
            use_learned_pos=bool(cfg.use_learned_pos),
            
            use_cam=bool(cfg.use_cam),
            use_ffn=bool(cfg.use_ffn),
            cam_tau=float(cfg.cam_tau),
            cam_init_scale=float(cfg.cam_init_scale),

        )

        print(
            f"[PTST] Train samples: {N} windows -> {N*D} CI samples | "
            f"L_eff={self.L_eff} patch_len={self.patch_len} n_patches={self.L_eff//self.patch_len} | "
            f"D={self.D} pred_len={self.pred_len} | d_model={self.d_model} heads={self.n_heads} depth={self.depth}"
        )

        rng = jax.random.PRNGKey(int(cfg.seed))
        dummy_x = jnp.zeros((1, self.L_eff, 1), dtype=jnp.float32)
        params = self.model.init(rng, dummy_x, deterministic=True)["params"]

        def create_state(lr: float):
            tx = optax.adamw(learning_rate=lr, weight_decay=0.0)
            return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        state = create_state(float(cfg.init_lr))
        self._rng = rng

        @jax.jit
        def train_step(state, xb, yb, rng):
            dropout_rng, new_rng = jax.random.split(rng)

            def loss_fn(p):
                yhat = state.apply_fn({"params": p}, xb, deterministic=False, rngs={"dropout": dropout_rng})
                loss = jnp.mean((yhat - yb) ** 2)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss, new_rng

        @jax.jit
        def eval_step(state, xb, yb):
            yhat = state.apply_fn({"params": state.params}, xb, deterministic=True)
            return jnp.mean((yhat - yb) ** 2)

        # move to device once
        Xtr_j = jnp.asarray(Xtr)
        Ytr_j = jnp.asarray(Ytr)
        Xva_j = jnp.asarray(Xva)
        Yva_j = jnp.asarray(Yva)

        best_params = state.params
        best_val = float("inf")
        curr_lr = float(cfg.init_lr)
        epochs_no_gain = 0

        def num_batches(n: int) -> int:
            return int(np.ceil(n / cfg.batch_size))

        # training loop
        for epoch in range(1, int(cfg.max_epochs) + 1):
            idx = np.arange(n_tr)
            np.random.default_rng(epoch + cfg.seed).shuffle(idx)

            train_losses = []
            rng = self._rng

            for bi in range(num_batches(n_tr)):
                s = bi * cfg.batch_size
                e = min((bi + 1) * cfg.batch_size, n_tr)
                bidx = idx[s:e]
                xb = Xtr_j[bidx]
                yb = Ytr_j[bidx]
                state, loss, rng = train_step(state, xb, yb, rng)
                train_losses.append(float(loss))

            self._rng = rng
            tr_loss = float(np.mean(train_losses))

            val_losses = []
            for bi in range(num_batches(n_val)):
                s = bi * cfg.batch_size
                e = min((bi + 1) * cfg.batch_size, n_val)
                xb = Xva_j[s:e]
                yb = Yva_j[s:e]
                val_losses.append(float(eval_step(state, xb, yb)))
            va_loss = float(np.mean(val_losses))

            print(f"[PTST] epoch {epoch:03d} | train {tr_loss:.6e} | val {va_loss:.6e} | lr {curr_lr:.2e}")

            if va_loss + 1e-8 < best_val:
                best_val = va_loss
                best_params = state.params
                epochs_no_gain = 0
            else:
                epochs_no_gain += 1

            # plateau schedule
            if epochs_no_gain >= int(cfg.patience):
                if curr_lr > float(cfg.min_lr) * (1.0 + 1e-9):
                    curr_lr = max(float(cfg.min_lr), curr_lr * float(cfg.lr_decay))
                    print(f"[PTST] plateau → lowering LR to {curr_lr:.2e}")
                    params_now = state.params
                    state = train_state.TrainState.create(
                        apply_fn=state.apply_fn,
                        params=params_now,
                        tx=optax.adamw(learning_rate=curr_lr, weight_decay=0.0),
                    )
                    epochs_no_gain = 0
                else:
                    print(f"[PTST] early stop: lr at min and no improvement (best val {best_val:.6e})")
                    break

        # store best
        self.state = state.replace(params=best_params)
        self.params = self.state.params

        # param count
        self.param_count = int(sum(p.size for p in jax.tree_util.tree_leaves(self.params)))
        print(f"[PTST] params: {self.param_count:,}")

    # ------------------------------------------------------------
    # Internal: one-shot predict (normalized space) for a batch
    # ------------------------------------------------------------
    def _predict_block_norm(self, ctx_norm_BLD: jnp.ndarray) -> jnp.ndarray:
        """
        ctx_norm_BLD: (B, L_eff, D) normalized
        returns: (B, pred_len, D) normalized
        """
        B, L, D = ctx_norm_BLD.shape
        assert L == self.L_eff
        assert D == self.D

        # channel-independent apply:
        # reshape -> (B*D, L, 1)
        x = ctx_norm_BLD.transpose(0, 2, 1)[:, :, :, None]  # (B, D, L, 1)
        x = x.reshape(B * D, L, 1)
        yhat = self.model.apply({"params": self.params}, x, deterministic=True)  # (B*D, pred_len)
        yhat = yhat.reshape(B, D, self.pred_len)  # (B,D,H)
        yhat = yhat.transpose(0, 2, 1)            # (B,H,D)
        return yhat

    # ------------------------------------------------------------
    # Public API: forecast
    # ------------------------------------------------------------
    def __call__(self, prefix: np.ndarray, steps: int) -> np.ndarray:
        """
        Forecast `steps` time points given prefix ending at forecast start.

        prefix: (ell, D)
        returns: (steps, D)
        """
        X = _as_2d(prefix)
        ell, D = X.shape
        if D != self.D:
            raise ValueError(f"PTST expected D={self.D}, got {D}.")

        H = int(steps)
        if H <= 0:
            return np.zeros((0, self.D), dtype=np.float32)

        # take last L_eff points (pad on left if needed)
        if ell < self.L_eff:
            pad_len = self.L_eff - ell
            pad = np.repeat(X[:1], repeats=pad_len, axis=0)
            ctx = np.concatenate([pad, X], axis=0)
        else:
            ctx = X[-self.L_eff:, :]

        # normalize context
        ctx_n = (ctx - self._mu) / self._sd  # (L_eff, D)

        preds_n_all = []
        n_done = 0
        
        while n_done < H:
            ctx_n_j = jnp.asarray(ctx_n[None, :, :])  # (1,L_eff,D)
            block_n = np.array(self._predict_block_norm(ctx_n_j)[0])  # (pred_len, D)
        
            remaining = H - n_done
            take = min(self.pred_len, remaining)
        
            preds_n_all.append(block_n[:take])
            n_done += take
        
            # roll context
            ctx_n = np.concatenate([ctx_n, block_n[:take]], axis=0)
            ctx_n = ctx_n[-self.L_eff:, :]


        preds_n = np.concatenate(preds_n_all, axis=0)  # (H, D)
        preds = preds_n * self._sd + self._mu
        return preds


__all__ = ["PTST", "PTSTConfig"]
