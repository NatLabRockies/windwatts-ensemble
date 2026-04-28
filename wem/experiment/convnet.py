"""1D Convolutional Neural Network for CDF-to-CDF bias correction.

Processes 3 input CDFs (HRRR, WTK, WTK-LED) as a 3-channel 1D signal and
outputs a corrected 101-point CDF with architecturally-enforced monotonicity
(softplus + cumsum).

All PyTorch code is isolated here. The ``torch`` import is lazy so that
``runner.py`` remains loadable without torch installed.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

from wem.utils.logging import log
from wem.utils.ml import balance_indices, fold_seed

# Lazy torch import — checked at call sites
torch = None
nn = None


def _ensure_torch():
    """Import torch lazily; raise helpful error if unavailable."""
    global torch, nn
    if torch is None:
        try:
            import torch as _torch
            import torch.nn as _nn

            torch = _torch
            nn = _nn
        except ImportError as e:
            raise SystemExit(
                "ConvNet experiment requires PyTorch. "
                "Install with: pip install 'wem[convnet]'"
            ) from e


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CDFDataset:
    """Torch Dataset wrapping CDF, auxiliary, and target arrays.

    Parameters
    ----------
    cdf : np.ndarray
        (N, n_channels, 101) float32 — input CDF channels.
    aux : np.ndarray
        (N, n_aux) float32 — auxiliary features.
    target : np.ndarray
        (N, 101) float32 — observed CDF targets.
    """

    def __init__(self, cdf: np.ndarray, aux: np.ndarray, target: np.ndarray):
        _ensure_torch()
        self.cdf = torch.from_numpy(cdf).float()
        self.aux = torch.from_numpy(aux).float()
        self.target = torch.from_numpy(target).float()

    def __len__(self) -> int:
        return self.cdf.shape[0]

    def __getitem__(self, idx):
        return self.cdf[idx], self.aux[idx], self.target[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CDFConvNet:
    """1D ConvNet: (3, 101) CDFs + aux features -> (101,) corrected CDF.

    Architecture::

        Input CDFs (n_ch, 101)
        -> [Conv1D -> BN -> ReLU] x n_conv_layers
        -> Global Avg Pool -> 128-dim
        -> Concat(aux_embed)
        -> FC(160 -> 128) -> ReLU -> Dropout
        -> FC(128 -> 64) -> ReLU -> Dropout
        -> FC(64 -> 101) -> softplus -> cumsum  (monotonic, non-negative)

    Parameters
    ----------
    n_aux_features : int
        Number of auxiliary input features (lat, lon, height_m, elevation_m, ...).
    dropout : float
        Dropout rate between FC layers.
    n_conv_layers : int
        Number of conv blocks (2 or 3).
    n_channels : int
        Number of input CDF channels (default 3).
    """

    def __init__(
        self,
        n_aux_features: int,
        dropout: float = 0.3,
        n_conv_layers: int = 3,
        n_channels: int = 3,
    ):
        _ensure_torch()
        # We build the actual nn.Module here
        self._module = _CDFConvNetModule(
            n_aux_features=n_aux_features,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            n_channels=n_channels,
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


def _build_conv_block(in_ch, out_ch):
    """Build a Conv1D -> BatchNorm1d -> ReLU block."""
    _ensure_torch()
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class _CDFConvNetModule(object):
    """Actual nn.Module implementation — created after torch is imported."""

    def __new__(cls, *args, **kwargs):
        _ensure_torch()

        # Dynamically create a proper nn.Module subclass
        class _Module(nn.Module):
            def __init__(
                self_inner,
                n_aux_features: int,
                dropout: float = 0.3,
                n_conv_layers: int = 3,
                n_channels: int = 3,
            ):
                super().__init__()

                # Conv backbone
                channels = [n_channels, 32, 64, 128][:n_conv_layers + 1]
                conv_layers = []
                for i in range(len(channels) - 1):
                    conv_layers.append(_build_conv_block(channels[i], channels[i + 1]))
                self_inner.conv = nn.Sequential(*conv_layers)
                conv_out_dim = channels[-1]

                # Aux embedding
                self_inner.aux_embed = nn.Sequential(
                    nn.Linear(n_aux_features, 32),
                    nn.ReLU(inplace=True),
                ) if n_aux_features > 0 else None
                aux_dim = 32 if n_aux_features > 0 else 0

                # FC head
                fc_in = conv_out_dim + aux_dim
                self_inner.fc = nn.Sequential(
                    nn.Linear(fc_in, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(64, 101),
                )

                self_inner.softplus = nn.Softplus()

            def forward(self_inner, cdf, aux):
                # cdf: (B, C, 101), aux: (B, n_aux)
                x = self_inner.conv(cdf)  # (B, conv_out, 101)
                x = x.mean(dim=2)  # global average pool -> (B, conv_out)

                if self_inner.aux_embed is not None:
                    a = self_inner.aux_embed(aux)  # (B, 32)
                    x = torch.cat([x, a], dim=1)  # (B, conv_out + 32)

                x = self_inner.fc(x)  # (B, 101)
                x = self_inner.softplus(x)  # non-negative increments
                x = torch.cumsum(x, dim=1)  # monotonically non-decreasing
                return x

        return _Module(*args, **kwargs)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _compute_norm_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from a 2D array."""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, fill_nan: float = 0.0) -> np.ndarray:
    """Apply (x - mean) / std normalization, replacing NaN with *fill_nan*."""
    result = ((arr - mean) / std).astype(np.float32)
    np.nan_to_num(result, nan=fill_nan, copy=False)
    return result


# ---------------------------------------------------------------------------
# Fold worker
# ---------------------------------------------------------------------------


def run_one_fold_convnet(
    sid: str,
    cdf_input: np.ndarray,
    aux_input: np.ndarray,
    targets: np.ndarray,
    station_ids: np.ndarray,
    is_gs: np.ndarray,
    nbr_map: Dict[str, Set[str]],
    seed: int,
    balance_strategy: str = "downsample",
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 30,
    val_frac: float = 0.2,
    device: str = "cpu",
    n_conv_layers: int = 3,
) -> Tuple[str, np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
    """Train a ConvNet for one LOOCV fold and predict the test station's CDF.

    Parameters
    ----------
    sid : str
        Station ID for the held-out test station.
    cdf_input : np.ndarray
        (N, n_channels, 101) CDF input array.
    aux_input : np.ndarray
        (N, n_aux) auxiliary feature array.
    targets : np.ndarray
        (N, 101) target CDF array.
    station_ids : np.ndarray
        (N,) station IDs.
    is_gs : np.ndarray
        (N,) boolean mask for Gold Standard stations.
    nbr_map : dict
        Neighbor map from ``build_neighbor_map``.
    seed : int
        Base random seed.
    balance_strategy : str
        ``"downsample"`` or ``"upsample"``.
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
    n_conv_layers : int
        Number of conv blocks (2 or 3).

    Returns
    -------
    tuple
        ``(sid, test_indices, predictions_101, metrics_or_None)``
    """
    _ensure_torch()
    n_quantiles = targets.shape[1]
    n_channels = cdf_input.shape[1]

    # Test rows = this GS station
    test_idx = np.where(station_ids == sid)[0]
    if test_idx.size == 0:
        return sid, np.array([], dtype=int), np.empty((0, n_quantiles), dtype=np.float32), None

    # Exclusion set: this sid + its 10km neighbors
    excl: Set[str] = {sid}
    if sid in nbr_map:
        excl |= set(nbr_map[sid])

    excl_arr = np.fromiter(excl, dtype=station_ids.dtype)
    base_train_mask = ~np.isin(station_ids, excl_arr) & np.all(np.isfinite(targets), axis=1)
    base_idx = np.where(base_train_mask)[0]
    if base_idx.size == 0:
        return sid, test_idx, np.full((test_idx.size, n_quantiles), np.nan, dtype=np.float32), None

    # Require finite CDF features (aux NaN handled by fill after normalization)
    finite_cdf = np.all(np.isfinite(cdf_input[base_idx].reshape(base_idx.size, -1)), axis=1)
    good_train_idx = base_idx[finite_cdf]
    if good_train_idx.size < 5:
        return sid, test_idx, np.full((test_idx.size, n_quantiles), np.nan, dtype=np.float32), None

    # Balance GS/ASOS (or use all data if strategy is "none")
    fseed = fold_seed(seed, sid)
    rng = np.random.default_rng(fseed)

    if balance_strategy == "none":
        train_idx = good_train_idx.copy()
    else:
        idx_asos = good_train_idx[~is_gs[good_train_idx]]
        idx_gs = good_train_idx[is_gs[good_train_idx]]
        train_idx = balance_indices(idx_asos, idx_gs, rng, strategy=balance_strategy)
    if train_idx.size < 5:
        return sid, test_idx, np.full((test_idx.size, n_quantiles), np.nan, dtype=np.float32), None

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

    # Compute normalization stats from training set
    cdf_flat_train = cdf_input[actual_train_idx].reshape(actual_train_idx.size, -1)
    cdf_mean, cdf_std = _compute_norm_stats(cdf_flat_train)
    aux_mean, aux_std = _compute_norm_stats(aux_input[actual_train_idx])

    def normalize_cdf(arr):
        flat = arr.reshape(arr.shape[0], -1)
        normed = _normalize(flat, cdf_mean, cdf_std)
        return normed.reshape(arr.shape)

    def normalize_aux(arr):
        return _normalize(arr, aux_mean, aux_std)

    # Prepare datasets
    cdf_train = normalize_cdf(cdf_input[actual_train_idx])
    aux_train = normalize_aux(aux_input[actual_train_idx])
    tgt_train = targets[actual_train_idx].astype(np.float32)
    train_ds = CDFDataset(cdf_train, aux_train, tgt_train)

    if use_early_stopping and val_idx is not None:
        cdf_val = normalize_cdf(cdf_input[val_idx])
        aux_val = normalize_aux(aux_input[val_idx])
        tgt_val = targets[val_idx].astype(np.float32)
        val_ds = CDFDataset(cdf_val, aux_val, tgt_val)

    cdf_test = normalize_cdf(cdf_input[test_idx])
    aux_test = normalize_aux(aux_input[test_idx])

    # Seed torch for reproducibility
    torch.manual_seed(fseed)

    # Build model
    dev = torch.device(device)
    n_aux = aux_input.shape[1]
    model = CDFConvNet(
        n_aux_features=n_aux,
        dropout=dropout,
        n_conv_layers=n_conv_layers,
        n_channels=n_channels,
    )
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(fseed),
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for cdf_b, aux_b, tgt_b in train_loader:
            cdf_b, aux_b, tgt_b = cdf_b.to(dev), aux_b.to(dev), tgt_b.to(dev)
            optimizer.zero_grad()
            out = model(cdf_b, aux_b)
            loss = loss_fn(out, tgt_b)
            loss.backward()
            optimizer.step()

        # Validation
        if use_early_stopping and val_idx is not None:
            model.eval()
            with torch.no_grad():
                val_cdf = val_ds.cdf.to(dev)
                val_aux = val_ds.aux.to(dev)
                val_tgt = val_ds.target.to(dev)
                val_out = model(val_cdf, val_aux)
                val_loss = loss_fn(val_out, val_tgt).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    # Load best weights if early stopping was used
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test
    model.eval()
    with torch.no_grad():
        test_cdf_t = torch.from_numpy(cdf_test).float().to(dev)
        test_aux_t = torch.from_numpy(aux_test).float().to(dev)
        preds_t = model(test_cdf_t, test_aux_t)
        preds = preds_t.cpu().numpy().astype(np.float32)

    # Compute metrics
    Y_test = targets[test_idx]
    good = np.isfinite(Y_test) & np.isfinite(preds)
    metrics = None
    if np.any(good):
        diff = preds[good] - Y_test[good]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        metrics = (rmse, mae)

    return sid, test_idx, preds, metrics
