from dataclasses import fields

import torch

from im2sim.configs.core import LayerConfig
from im2sim.configs.unet import UNetConfig
from im2sim.layers.image_conv_blocks import ImageConvBlock
from im2sim.utils.layer_util import (
    apply_residual_connection,
    call_with_supported_kwargs,
    get_image_layer,
)


class UNet(torch.nn.Module):
    """
    A flexible U-Net [1] implementation for image segmentation and reconstruction tasks.

    Args:
        in_channels (int):
            Number of input channels.

        out_channels (int):
            Number of output channels.

        rank (int):
            Spatial rank (1D, 2D, 3D).

        cfg (UNetConfig):
            Configuration object for the U-Net.

        supervision_levels (int | list[int]):
            Levels at which to apply deep supervision. `0` corresponds to the highest resolution output, `1` to the next lower resolution, and so on.
            Default is `0` (no deep supervision).

    Examples:

        To create a UNet model with a specific configuration, you can first create a UNetConfig object and then pass it to the UNet constructor.
        For example, to create a UNet with 3 levels of depth, ReLU activation, and softmax output activation:

        >>> cfg = UNetConfig(filters=[32, 64, 128],
        >>>            encoder_block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU"),
        >>>            decoder_block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU"),
        >>>            out_block_cfg=ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="softmax"))

        Since the configs are rankless, you could use the same config for a 1D, 2D, or 3D convolutional block by changing the rank parameter when creating the UNet instance.

        >>> model1D = UNet(
        >>>        rank=1,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model2D = UNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model3D = UNet(
        >>>        rank=3,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )

        The UNet model can be used for both segmentation and reconstruction tasks.
        For segmentation, you can use the `single_class_segmentation_mode()` or `multiclass_segmentation_mode()` methods of the UNetConfig to set the appropriate output activation function (sigmoid for single-class, softmax for multi-class).
        For reconstruction tasks, you can use the `reconstruction_mode()` method to set the output activation to None.

        >>> cfg_segmentation = UNetConfig(filters=[32, 64, 128]).single_class_segmentation_mode()
        >>> model_segmentation = UNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_segmentation,
        >>>    )

        >>> cfg_reconstruction = UNetConfig(filters=[32, 64, 128]).reconstruction_mode()
        >>> model_reconstruction = UNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_reconstruction,
        >>>    )

        If deep supervision is desired, you can specify the levels at which to apply it using the `supervision_levels` argument.

        >>> model_deep_supervision = UNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>        supervision_levels=[0, 1],  # Apply deep supervision at top 2 levels
        >>>    )

        Models can be saved and loaded using the standard PyTorch methods:

        >>> torch.save(model.state_dict(), "model.pth")
        >>> model.load_state_dict(torch.load("model.pth"))

        Configs can also be saved and loaded using the methods provided in the `im2sim.configs.UNetConfig` class:

        >>> cfg.save("my_config.yaml")
        >>> loaded_cfg = UNetConfig.load("my_config.yaml")
        >>> model = UNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=loaded_cfg,
        >>>    )


    References:
        .. [1] O. Ronneberger, P. Fischer, and T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation,
            May 18, 2015, arXiv: arXiv:1505.04597. doi: 10.48550/arXiv.1505.04597.


    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        cfg: UNetConfig,
        supervision_levels: int | list[int] = 0,
    ):
        """ """
        super().__init__()

        filters = cfg.filters
        self.L = len(filters)

        # ---- pooling / upsampling ----
        pool_cfg = cfg.pool_cfg
        upsample_cfg = cfg.upsample_cfg

        self.pools = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()

        # ---- encoder ----
        self.encoders = torch.nn.ModuleList()
        self.skip_blocks = torch.nn.ModuleList()
        in_ch = in_channels
        for i in range(self.L):
            out_ch = filters[i]

            block = torch.nn.Sequential(
                *[
                    ImageConvBlock(
                        in_channels=in_ch if j == 0 else out_ch,
                        out_channels=out_ch,
                        rank=rank,
                        cfg=cfg.encoder_block_cfg[i],
                    )
                    for j in range(cfg.encoder_blocks_per_level)
                ]
            )

            self.encoders.append(block)

            if i < self.L - 1:
                skip = ImageConvBlock(
                    in_channels=out_ch, out_channels=out_ch, rank=rank, cfg=cfg.skip_connection_cfg
                )
                self.skip_blocks.append(skip)

            if i < self.L - 1:
                # we need to pass in the in_channels and out_channels to the pooling layer, as some pooling layers (e.g., strided conv) require them
                pool = call_with_supported_kwargs(
                    get_image_layer(pool_cfg[i].name, rank),
                    {"in_channels": out_ch, "out_channels": out_ch, **pool_cfg[i].kwargs},
                )
                self.pools.append(pool)

            in_ch = out_ch

        if isinstance(supervision_levels, int):
            supervision_levels = [supervision_levels]

        self.supervision_levels = supervision_levels

        # ---- decoder ----
        self.decoders = torch.nn.ModuleList()
        self.out_blocks = torch.nn.ModuleList()

        for i in reversed(range(self.L - 1)):
            in_ch = filters[i]

            # upsample layer needs to know the number of channels in the input and output, as some upsampling layers (e.g., transposed conv) require them
            up = call_with_supported_kwargs(
                get_image_layer(upsample_cfg[i].name, rank),
                {
                    "in_channels": filters[i + 1],
                    "out_channels": filters[i + 1],
                    **upsample_cfg[i].kwargs,
                },
            )

            self.ups.append(up)

            if cfg.fusion_type.strip().lower() == "concat":
                in_ch += filters[i + 1]

            out_ch = filters[i]

            block = torch.nn.Sequential(
                *[
                    ImageConvBlock(
                        in_channels=in_ch if j == 0 else out_ch,
                        out_channels=out_ch,
                        rank=rank,
                        cfg=cfg.decoder_block_cfg[i],
                    )
                    for j in range(cfg.decoder_blocks_per_level)
                ]
            )

            self.decoders.append(block)

            # We only add output blocks for the levels specified in supervision_levels
            if i in supervision_levels:
                self.out_blocks.append(
                    ImageConvBlock(
                        in_channels=out_ch,
                        out_channels=out_channels,
                        rank=rank,
                        cfg=cfg.out_block_cfg,
                    )
                )

        self.fusion_type = cfg.fusion_type

    def forward(self, x):
        """
        Forward pass through the U-Net.
        """

        skips = []

        # ---- encoder ----
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.pools):
                skips.append(self.skip_blocks[i](x))
                x = self.pools[i](x)

        # ---- decoder ----
        decoder_outputs = []
        ctr = 0
        for i, (up, dec) in enumerate(zip(self.ups, self.decoders, strict=True)):
            x = up(x)
            skip = skips[-(i + 1)]

            # dynamic shape alignment
            if x.shape[2:] != skip.shape[2:]:
                skip = torch.nn.functional.interpolate(skip, size=x.shape[2:])

            x = apply_residual_connection(x, skip, connection_type=self.fusion_type)
            x = dec(x)

            if (self.L - i - 2) in self.supervision_levels:
                decoder_outputs.append(self.out_blocks[ctr](x))
                ctr += 1

        # get deep supervision outputs and upsample if required
        out_shape = decoder_outputs[-1].shape[2:]
        for i in range(len(decoder_outputs)):
            if decoder_outputs[i].shape[2:] != out_shape:
                decoder_outputs[i] = torch.nn.functional.interpolate(
                    decoder_outputs[i], size=out_shape
                )

        if len(decoder_outputs) > 1:
            return decoder_outputs

        # if only one output, return it directly
        return decoder_outputs[-1]


if __name__ == "__main__":

    def cfg_print(cfg):
        print("###################")
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            print(f"  {field.name}: {value}")

    pool_cfg = [
        LayerConfig(name="MaxPool", kwargs={"kernel_size": (1, 2, 2)}),
        LayerConfig(name="AvgPool", kwargs={"kernel_size": 2}),
        LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
    ]
    upsample_cfg = [
        LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
        LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
        LayerConfig(name="Upsample", kwargs={"scale_factor": (1, 2, 2), "mode": "bilinear"}),
    ]
    cfg = UNetConfig(filters=[32, 64, 128, 256], pool_cfg=pool_cfg, upsample_cfg=upsample_cfg)

    model1D = UNet(
        rank=1,
        in_channels=32,
        out_channels=32,
        cfg=cfg,
    )
    model2D = UNet(
        rank=2,
        in_channels=32,
        out_channels=32,
        cfg=cfg,
    )
    model3D = UNet(
        rank=3,
        in_channels=32,
        out_channels=32,
        cfg=cfg,
    )

    total_params = sum(p.numel() for p in model3D.parameters())
    print(f"Parameters: {total_params:,}")

    print(model3D)

    x = torch.randn(1, 32, 64, 64, 64)  # Example input for a 3D tensor

    def check_gradients(model, x):
        model.train()

        # Ensure input tracks gradients if you want to test input gradients
        x = x.requires_grad_(True)

        print(x.shape)
        # Forward
        y = model(x)
        print(y[-1].shape)

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
    if check_gradients(model3D, x):
        print("Gradient check passed.")
