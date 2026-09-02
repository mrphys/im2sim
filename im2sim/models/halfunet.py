from dataclasses import fields

import torch

from im2sim.configs.core import LayerConfig
from im2sim.configs.halfunet import HalfUNetConfig
from im2sim.layers.image_blocks import ImageConvBlock
from im2sim.utils.layer_util import (
    apply_residual_connection,
    call_with_supported_kwargs,
    get_image_layer,
)


class HalfUNet(torch.nn.Module):
    """
    A flexible HalfUNet [1] implementation for image segmentation and reconstruction tasks.

    Args:
        in_channels (int):
            Number of input channels.

        out_channels (int):
            Number of output channels.

        rank (int):
            Spatial rank (1D, 2D, 3D).

        cfg (HalfUNetConfig):
            Configuration object for the HalfUNet.

        supervision_levels (int | list[int]):
            Levels at which to apply deep supervision. `0` corresponds to the highest resolution output, `1` to the next lower resolution, and so on.
            Default is `0`(no deep supervision).

    Examples:

        To create a HalfUNet model with a specific configuration, you can first create a HalfUNetConfig object and then pass it to the HalfUNet constructor.
        For example, to create a HalfUNet with 3 levels of depth, ReLU activation, and softmax output activation:

        >>> cfg = HalfUNetConfig(
        >>>            hidden_channels=32,
        >>>            n_levels=3,
        >>>            encoder_block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU"),
        >>>            out_block_cfg=ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="softmax")
        >>>       )

        Since the configs are rankless, you could use the same config for a 1D, 2D, or 3D convolutional block by changing the rank parameter when creating the HalfUNet instance.

        >>> model1D = HalfUNet(
        >>>        rank=1,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model2D = HalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model3D = HalfUNet(
        >>>        rank=3,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )

        The HalfUNet model can be used for both segmentation and reconstruction tasks.
        For segmentation, you can use the `single_class_segmentation_mode()` or `multiclass_segmentation_mode()` methods of the HalfUNetConfig to set the appropriate output activation function (sigmoid for single-class, softmax for multi-class).
        For reconstruction tasks, you can use the `reconstruction_mode()` method to set the output activation to None.

        >>> cfg_segmentation = HalfUNetConfig().single_class_segmentation_mode()
        >>> model_segmentation = HalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_segmentation,
        >>>    )

        >>> cfg_reconstruction = HalfUNetConfig().reconstruction_mode()
        >>> model_reconstruction = HalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_reconstruction,
        >>>    )


        Models can be saved and loaded using the standard PyTorch methods:

        >>> torch.save(model.state_dict(), "model.pth")
        >>> model.load_state_dict(torch.load("model.pth"))

        Configs can also be saved and loaded using the methods provided in the `im2sim.configs.UNetConfig` class:

        >>> cfg.save("my_config.yaml")
        >>> loaded_cfg = HalfUNetConfig.load("my_config.yaml")
        >>> model = HalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=loaded_cfg,
        >>>    )


    References:
        .. [1] H. Lu, Y. She, J. Tie, and S. Xu, Half-UNet: A Simplified HalfUNet Architecture for Medical Image Segmentation,
            Front. Neuroinformatics, vol. 16, Jun. 2022, doi: 10.3389/fninf.2022.911679.


    """

    def __init__(self, in_channels: int, out_channels: int, rank: int, cfg: HalfUNetConfig):
        """ """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.n_levels = cfg.n_levels
        self.hidden_channels = cfg.hidden_channels

        # ---- pooling / upsampling ----
        pool_cfg = cfg.pool_cfg
        upsample_cfg = cfg.upsample_cfg

        self.pools = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()

        # ---- stem ----

        self.stem = ImageConvBlock(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            rank=rank,
            cfg=cfg.stem_block_cfg,
        )

        # ---- encoder ----
        self.encoders = torch.nn.ModuleList()
        for i in range(self.n_levels):
            block = torch.nn.Sequential(
                *[
                    ImageConvBlock(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        rank=rank,
                        cfg=cfg.encoder_block_cfg[i],
                    )
                    for _ in range(cfg.encoder_blocks_per_level)
                ]
            )

            self.encoders.append(block)

            if i < self.n_levels - 1:
                # we need to pass in the in_channels and out_channels to the pooling layer, as some pooling layers (e.g., strided conv) require them
                pool = call_with_supported_kwargs(
                    get_image_layer(pool_cfg[i].name, rank),
                    {
                        "in_channels": self.hidden_channels,
                        "out_channels": self.hidden_channels,
                        **pool_cfg[i].kwargs,
                    },
                )
                self.pools.append(pool)

        # ---- upsampling ----
        for i in reversed(range(self.n_levels - 1)):
            channels = (
                self.hidden_channels
                if cfg.fusion_type == "add"
                else self.hidden_channels * (self.n_levels - i - 1)
            )
            up = call_with_supported_kwargs(
                get_image_layer(upsample_cfg[i].name, rank),
                {"in_channels": channels, "out_channels": channels, **upsample_cfg[i].kwargs},
            )
            self.ups.append(up)

        # ---- out_conv ----
        self.fusion_type = cfg.fusion_type

        self.out_block = ImageConvBlock(
            in_channels=self.hidden_channels
            if self.fusion_type == "add"
            else self.hidden_channels * self.n_levels,
            out_channels=out_channels,
            rank=rank,
            cfg=cfg.out_block_cfg,
        )

    def forward(self, x):
        """
        Forward pass through the HalfUNet.
        """

        x = self.stem(x)
        enc_outputs = []
        # ---- encoder ----
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            enc_outputs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        # ---- fusion ----
        for up, enc in zip(self.ups, reversed(enc_outputs[:-1]), strict=True):
            x = up(x)
            x = apply_residual_connection(x, enc, connection_type=self.fusion_type)

        output = self.out_block(x)
        return output


if __name__ == "__main__":

    def cfg_print(cfg):
        print("###################")
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            print(f"  {field.name}: {value}")

    cfg = HalfUNetConfig(
        upsample_cfg=LayerConfig(name="ConvTranspose", kwargs={"kernel_size": 2, "stride": 2}),
        fusion_type="concat",
    )

    # cfg = cfg.apply_presets(["single_class_segmentation", "residual", "SE", "ghost_depthwise_separable"])
    model = HalfUNet(rank=2, in_channels=20, out_channels=1, cfg=cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print(model)

    x = torch.randn(1, 20, 64, 64)  # Example input for a 3D tensor

    def check_gradients(model, x):
        model.train()

        # Ensure input tracks gradients if you want to test input gradients
        x = x.requires_grad_(True)

        print(x.shape)
        # Forward
        y = model(x)
        print(y.shape)

        # Use a scalar loss
        loss = sum([output.sum() for output in y]) if isinstance(y, list) else y.sum()

        # Backward
        loss.backward()

        # Check input gradient
        assert x.grad is not None, "Input gradient is None"
        assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"

        # Check parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} gradient is None"
                assert torch.isfinite(param.grad).all(), f"{name} gradient contains NaN/Inf"
                assert param.grad.abs().sum() > 0, f"{name} gradient is zero"

        return True

    # Check gradients
    if check_gradients(model, x):
        print("Gradient check passed.")
