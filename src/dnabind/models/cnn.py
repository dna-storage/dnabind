"""Configurable CNN model families.

Two families, both driven entirely by an architecture JSON:

* ``cnn_4xn`` — for 4 x (2L) one-hot style inputs (e.g. the ``onehot_interleaved``
  encoder). A 2D conv collapses the height-4 axis, then a 1D conv stack runs
  along the length axis, then a linear head.
* ``cnn_nxn`` — for square L x L pair-matrix inputs (e.g. the ``wc_binary``
  encoder). A 2D conv stack over the L x L grid, then a linear head.

Both forward passes return ``(logits, preds)`` with ``preds = sigmoid(logits)``.
Input dimensions come from the encoder via ``from_config(config, input_shape)``,
so the JSON only describes layers, never the input size.
"""

import torch
import torch.nn as nn

from .registry import register_model

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _activation(config):
    name = config.get("activation")
    if not name:
        return None
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}")
    return _ACTIVATIONS[name]()


@register_model("cnn_4xn")
class CNN4xN(nn.Module):
    """2D conv (collapses the 4-row axis) -> 1D conv stack -> linear head."""

    def __init__(
        self,
        input_channels,
        input_matrix_width,
        conv2d_layers_config,
        conv1d_layers_config=None,
        linear_layers_config=None,
        conv1d_final_global_pool=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_matrix_width = input_matrix_width

        # --- 2D conv block ---
        modules = []
        in_ch = input_channels
        for cfg in conv2d_layers_config:
            modules.append(
                nn.Conv2d(
                    in_ch,
                    cfg["out_channels"],
                    kernel_size=cfg["kernel_size"],
                    stride=cfg.get("stride", 1),
                    padding=cfg.get("padding", 0),
                )
            )
            if cfg.get("batch_norm"):
                modules.append(nn.BatchNorm2d(cfg["out_channels"]))
            act = _activation(cfg)
            if act:
                modules.append(act)
            if cfg.get("dropout", 0) > 0:
                modules.append(nn.Dropout2d(cfg["dropout"]))
            in_ch = cfg["out_channels"]
        self.conv2d_block = nn.Sequential(*modules)

        # --- 1D conv block ---
        modules = []
        if conv1d_layers_config:
            for cfg in conv1d_layers_config:
                modules.append(
                    nn.Conv1d(
                        in_ch,
                        cfg["out_channels"],
                        kernel_size=cfg["kernel_size"],
                        stride=cfg.get("stride", 1),
                        padding=cfg.get("padding", 0),
                    )
                )
                if cfg.get("batch_norm"):
                    modules.append(nn.BatchNorm1d(cfg["out_channels"]))
                act = _activation(cfg)
                if act:
                    modules.append(act)
                if cfg.get("dropout", 0) > 0:
                    modules.append(nn.Dropout(cfg["dropout"]))
                in_ch = cfg["out_channels"]
        if conv1d_final_global_pool is not None:
            gp = conv1d_final_global_pool.lower()
            if gp == "avg":
                modules.append(nn.AdaptiveAvgPool1d(1))
            elif gp == "max":
                modules.append(nn.AdaptiveMaxPool1d(1))
            else:
                raise ValueError(
                    f"Unsupported conv1d_final_global_pool: {conv1d_final_global_pool!r}"
                )
        self.conv1d_block = nn.Sequential(*modules) if modules else nn.Identity()

        # --- linear head (flattened size inferred from a dummy pass) ---
        self.linear_block = self._build_linear_head(linear_layers_config)
        self.sigmoid = nn.Sigmoid()

    def _build_linear_head(self, linear_layers_config):
        if not linear_layers_config:
            return nn.Identity()
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, 4, self.input_matrix_width)
            out = self.conv2d_block(dummy).squeeze(2)
            out = self.conv1d_block(out)
            in_features = out.view(1, -1).shape[1]
        modules = []
        for cfg in linear_layers_config:
            modules.append(nn.Linear(in_features, cfg["out_features"]))
            if cfg.get("batch_norm"):
                modules.append(nn.BatchNorm1d(cfg["out_features"]))
            act = _activation(cfg)
            if act:
                modules.append(act)
            if cfg.get("dropout", 0) > 0:
                modules.append(nn.Dropout(cfg["dropout"]))
            in_features = cfg["out_features"]
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv2d_block(x)
        x = x.squeeze(2)  # collapse the height-4 axis
        x = self.conv1d_block(x)
        x = x.view(x.size(0), -1)
        logits = self.linear_block(x)
        return logits, self.sigmoid(logits)

    @classmethod
    def from_config(cls, config, input_shape):
        channels, height, width = input_shape
        if height != 4:
            raise ValueError(
                f"cnn_4xn expects an encoder with height 4; got input_shape={input_shape}."
            )
        return cls(
            input_channels=channels,
            input_matrix_width=width,
            conv2d_layers_config=config["conv2d_layers_config"],
            conv1d_layers_config=config.get("conv1d_layers_config"),
            linear_layers_config=config.get("linear_layers_config"),
            conv1d_final_global_pool=config.get("conv1d_final_global_pool"),
        )


@register_model("cnn_nxn")
class CNNNxN(nn.Module):
    """2D conv stack over a square L x L input -> linear head."""

    def __init__(
        self,
        input_channels,
        input_size,
        conv2d_layers_config,
        linear_layers_config=None,
        final_global_pool=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size

        modules = []
        in_ch = input_channels
        for cfg in conv2d_layers_config:
            modules.append(
                nn.Conv2d(
                    in_ch,
                    cfg["out_channels"],
                    kernel_size=cfg["kernel_size"],
                    stride=cfg.get("stride", 1),
                    padding=cfg.get("padding", 0),
                )
            )
            if cfg.get("batch_norm"):
                modules.append(nn.BatchNorm2d(cfg["out_channels"]))
            act = _activation(cfg)
            if act:
                modules.append(act)
            pool = cfg.get("pool")
            if pool:
                size = pool.get("size", 2)
                ptype = pool.get("type", "max").lower()
                if ptype == "max":
                    modules.append(nn.MaxPool2d(size))
                elif ptype == "avg":
                    modules.append(nn.AvgPool2d(size))
                else:
                    raise ValueError(f"Unsupported pool type: {ptype}")
            if cfg.get("dropout", 0) > 0:
                modules.append(nn.Dropout2d(cfg["dropout"]))
            in_ch = cfg["out_channels"]
        if final_global_pool is not None:
            gp = final_global_pool.lower()
            if gp == "avg":
                modules.append(nn.AdaptiveAvgPool2d(1))
            elif gp == "max":
                modules.append(nn.AdaptiveMaxPool2d(1))
            else:
                raise ValueError(f"Unsupported final_global_pool: {final_global_pool!r}")
        self.conv2d_block = nn.Sequential(*modules)

        self.linear_block = self._build_linear_head(linear_layers_config)
        self.sigmoid = nn.Sigmoid()

    def _build_linear_head(self, linear_layers_config):
        if not linear_layers_config:
            return nn.Identity()
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.input_channels, self.input_size, self.input_size
            )
            out = self.conv2d_block(dummy)
            in_features = out.view(1, -1).shape[1]
        modules = []
        for cfg in linear_layers_config:
            modules.append(nn.Linear(in_features, cfg["out_features"]))
            if cfg.get("batch_norm"):
                modules.append(nn.BatchNorm1d(cfg["out_features"]))
            act = _activation(cfg)
            if act:
                modules.append(act)
            if cfg.get("dropout", 0) > 0:
                modules.append(nn.Dropout(cfg["dropout"]))
            in_features = cfg["out_features"]
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv2d_block(x)
        x = x.view(x.size(0), -1)
        logits = self.linear_block(x)
        return logits, self.sigmoid(logits)

    @classmethod
    def from_config(cls, config, input_shape):
        channels, height, width = input_shape
        if height != width:
            raise ValueError(
                f"cnn_nxn expects a square input; got input_shape={input_shape}."
            )
        return cls(
            input_channels=channels,
            input_size=height,
            conv2d_layers_config=config["conv2d_layers_config"],
            linear_layers_config=config.get("linear_layers_config"),
            final_global_pool=config.get("final_global_pool"),
        )
