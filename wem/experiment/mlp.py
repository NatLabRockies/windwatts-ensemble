"""Long-format MLP for tabular wind speed bias correction.

Same 9 features and ~221K training rows as the baseline XGBoost, but uses a
two-hidden-layer MLP instead of gradient-boosted trees.  All PyTorch code is
isolated here; the ``torch`` import is lazy so ``runner.py`` remains loadable
without torch installed.

Architecture::

    Input (9) → Linear(128) → ReLU → Dropout(0.3)
             → Linear(64)  → ReLU → Dropout(0.3)
             → Linear(1)

Training uses MAE loss (matching XGBoost ``reg:absoluteerror``), AdamW with
weight decay, and ``ReduceLROnPlateau`` with station-level early stopping.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

from wem.experiment.convnet import _compute_norm_stats, _ensure_torch, _normalize
from wem.utils.logging import log
from wem.utils.ml import balance_indices, fold_seed

# Lazy torch references — populated by _ensure_torch()
torch = None
nn = None


def _ensure_mlp_torch():
    """Import torch lazily and set module-level references."""
    global torch, nn
    _ensure_torch()
    import torch as _torch
    import torch.nn as _nn

    torch = _torch
    nn = _nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TabularMLP:
    """Simple feedforward MLP for tabular regression.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dims : tuple[int, ...]
        Sizes of hidden layers (default ``(128, 64)``).
    dropout : float
        Dropout rate between hidden layers.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
    ):
        _ensure_mlp_torch()
        self._module = _TabularMLPModule(
            n_features=n_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    @property
    def module(self):
        return self._module

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def parameters(self):
        return self._module.parameters()

    def train(self, mode=True):
        return self._module.train(mode)

    def eval(self):
        return self._module.eval()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self._module.load_state_dict(*args, **kwargs)

    def to(self, device):
        self._module = self._module.to(device)
        return self


class _TabularMLPModule(object):
    """Actual nn.Module — created after torch is imported."""

    def __new__(cls, *args, **kwargs):
        _ensure_mlp_torch()

        class _Module(nn.Module):
            def __init__(
                self_inner,
                n_features: int,
                hidden_dims: tuple[int, ...] = (128, 64),
                dropout: float = 0.3,
            ):
                super().__init__()
                layers = []
                in_dim = n_features
                for h in hidden_dims:
                    layers.append(nn.Linear(in_dim, h))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout))
                    in_dim = h
                layers.append(nn.Linear(in_dim, 1))
                self_inner.net = nn.Sequential(*layers)

            def forward(self_inner, x):
                # x: (B, n_features) → (B, 1)
                return self_inner.net(x)

        return _Module(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fold worker
# ---------------------------------------------------------------------------


def run_one_fold_mlp(
    sid: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    station_ids: np.ndarray,
    is_gs: np.ndarray,
    nbr_map: Dict[str, Set[str]],
    seed: int,
    balance_strategy: str = "downsample",
    require_finite_mask: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 15,
    val_frac: float = 0.2,
    device: str = "cpu",
    hidden_dims: tuple[int, ...] = (128, 64),
) -> Tuple[str, np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
    """Train an MLP for one LOOCV fold and predict the test station.

    Parameters
    ----------
    sid : str
        Station ID for the held-out test station.
    X_full : np.ndarray
        (N, F) feature array (same as baseline XGBoost).
    y_full : np.ndarray
        (N,) target array (observation wind speed in m/s).
    station_ids : np.ndarray
        (N,) station IDs.
    is_gs : np.ndarray
        (N,) boolean mask for Gold Standard stations.
    nbr_map : dict
        Neighbor map from ``build_neighbor_map``.
    seed : int
        Base random seed.
    balance_strategy : str
        ``"downsample"``, ``"upsample"``, or ``"none"``.
    require_finite_mask : np.ndarray or None
        Boolean mask over feature columns; True = must be finite.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    dropout : float
        Dropout rate.
    patience : int
        Early stopping patience (epochs without val improvement).
    val_frac : float
        Fraction of training stations held out for validation.
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.
    hidden_dims : tuple[int, ...]
        Hidden layer sizes (default ``(128, 64)``).

    Returns
    -------
    tuple
        ``(sid, test_indices, predictions_1d, metrics_or_None)``
    """
    _ensure_mlp_torch()

    # Test rows = this GS station
    test_idx = np.where(station_ids == sid)[0]
    if test_idx.size == 0:
        return sid, np.array([], dtype=int), np.empty(0, dtype=np.float32), None

    # Exclusion set: this sid + its 10km neighbors
    excl: Set[str] = {sid}
    if sid in nbr_map:
        excl |= set(nbr_map[sid])

    excl_arr = np.fromiter(excl, dtype=station_ids.dtype)
    base_train_mask = ~np.isin(station_ids, excl_arr) & np.isfinite(y_full)
    base_idx = np.where(base_train_mask)[0]
    if base_idx.size == 0:
        return sid, test_idx, np.full(test_idx.size, np.nan, dtype=np.float32), None

    # Require finite features (same logic as XGBoost baseline)
    if require_finite_mask is not None:
        finite_cols = X_full[base_idx][:, require_finite_mask]
        finite_rows = np.all(np.isfinite(finite_cols), axis=1)
    else:
        finite_rows = np.all(np.isfinite(X_full[base_idx]), axis=1)
    good_train_idx = base_idx[finite_rows]
    if good_train_idx.size < 5:
        return sid, test_idx, np.full(test_idx.size, np.nan, dtype=np.float32), None

    # Balance GS/ASOS
    fseed = fold_seed(seed, sid)
    rng = np.random.default_rng(fseed)

    if balance_strategy == "none":
        train_idx = good_train_idx.copy()
    else:
        idx_asos = good_train_idx[~is_gs[good_train_idx]]
        idx_gs = good_train_idx[is_gs[good_train_idx]]
        train_idx = balance_indices(idx_asos, idx_gs, rng, strategy=balance_strategy)
    if train_idx.size < 5:
        return sid, test_idx, np.full(test_idx.size, np.nan, dtype=np.float32), None

    rng.shuffle(train_idx)

    # Station-level train/val split
    train_sids = np.unique(station_ids[train_idx])
    use_early_stopping = len(train_sids) >= 5 and val_frac > 0
    if use_early_stopping:
        n_val_stations = max(1, int(len(train_sids) * val_frac))
        rng.shuffle(train_sids)
        val_station_set = set(train_sids[:n_val_stations])

        val_mask = np.isin(station_ids[train_idx], list(val_station_set))
        val_idx = train_idx[val_mask]
        actual_train_idx = train_idx[~val_mask]

        if actual_train_idx.size < 3 or val_idx.size < 1:
            use_early_stopping = False
            actual_train_idx = train_idx
            val_idx = None
    else:
        actual_train_idx = train_idx
        val_idx = None

    # Compute normalization stats from training features
    feat_mean, feat_std = _compute_norm_stats(X_full[actual_train_idx])

    X_train = _normalize(X_full[actual_train_idx], feat_mean, feat_std, fill_nan=0.0)
    y_train = y_full[actual_train_idx].astype(np.float32)

    if use_early_stopping and val_idx is not None:
        X_val = _normalize(X_full[val_idx], feat_mean, feat_std, fill_nan=0.0)
        y_val = y_full[val_idx].astype(np.float32)

    X_test = _normalize(X_full[test_idx], feat_mean, feat_std, fill_nan=0.0)

    # Seed torch for reproducibility
    torch.manual_seed(fseed)

    # Build model
    dev = torch.device(device)
    n_features = X_full.shape[1]
    model = TabularMLP(
        n_features=n_features,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    train_t = torch.from_numpy(X_train).float()
    target_t = torch.from_numpy(y_train).float().unsqueeze(1)  # (N, 1)
    train_ds = torch.utils.data.TensorDataset(train_t, target_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(fseed),
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

        # Validation
        if use_early_stopping and val_idx is not None:
            model.eval()
            with torch.no_grad():
                val_x = torch.from_numpy(X_val).float().to(dev)
                val_y = torch.from_numpy(y_val).float().unsqueeze(1).to(dev)
                val_out = model(val_x)
                val_loss = loss_fn(val_out, val_y).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log(f"  Early stop at epoch {epoch + 1}/{epochs} "
                        f"(best={best_epoch + 1}, val_mae={best_val_loss:.4f})")
                    break
    else:
        # Ran all epochs without early stopping trigger
        if use_early_stopping and val_idx is not None:
            log(f"  Ran all {epochs} epochs (best={best_epoch + 1}, val_mae={best_val_loss:.4f})")

    # Load best weights if early stopping was used
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test
    model.eval()
    with torch.no_grad():
        test_x = torch.from_numpy(X_test).float().to(dev)
        preds_t = model(test_x)
        preds = preds_t.squeeze(1).cpu().numpy().astype(np.float32)

    # Compute metrics
    Y_test = y_full[test_idx]
    good = np.isfinite(Y_test) & np.isfinite(preds)
    metrics = None
    if np.any(good):
        diff = preds[good] - Y_test[good]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        metrics = (rmse, mae)

    return sid, test_idx, preds, metrics
