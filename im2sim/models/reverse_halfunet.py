from dataclasses import fields

import torch

from im2sim.configs.reverse_halfunet import ReverseHalfUNetConfig
from im2sim.layers.image_blocks import ImageConvBlock
from im2sim.utils.layer_util import (
    apply_residual_connection,
    call_with_supported_kwargs,
    get_image_layer,
)


class ReverseHalfUNet(torch.nn.Module):
    """
    A flexible implementation of a 'Reverse HalfUNet', which is inspired by a HalfUNet[1] architecture but
    where the HalfUNet does away with the decoders of a conventional UNet[2], this model instead removes the encoders.
    Can be used for image segmentation and reconstruction tasks where blurring caused by additive fusion as in a HalfUNet is undesirable and/or deep supervision is required.

    Args:
        in_channels (int):
            Number of input channels.

        out_channels (int):
            Number of output channels.

        rank (int):
            Spatial rank (1D, 2D, 3D).

        cfg (ReverseHalfUNetConfig):
            Configuration object for the Reverse ReverseHalfUNet.

        supervision_levels (int | list[int]):
            Levels at which to apply deep supervision. `0` corresponds to the highest resolution output, `1` to the next lower resolution, and so on.
            Default is `0` (no deep supervision).

    Examples:

        To create a ReverseHalfUNet model with a specific configuration, you can first create a ReverseHalfUNetConfig object and then pass it to the ReverseHalfUNet constructor.
        For example, to create a ReverseHalfUNet with 3 levels of depth, ReLU activation, and softmax output activation:

        >>> cfg = ReverseHalfUNetConfig(
        >>>            hidden_channels=32,
        >>>            n_levels=3,
        >>>            decoder_block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU"),
        >>>            out_block_cfg=ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="softmax")
        >>>       )

        Since the configs are rankless, you could use the same config for a 1D, 2D, or 3D convolutional block by changing the rank parameter when creating the ReverseHalfUNet instance.

        >>> model1D = ReverseHalfUNet(
        >>>        rank=1,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model2D = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model3D = ReverseHalfUNet(
        >>>        rank=3,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )

        The ReverseHalfUNet model can be used for both segmentation and reconstruction tasks.
        For segmentation, you can use the `single_class_segmentation_mode()` or `multiclass_segmentation_mode()` methods of the ReverseHalfUNetConfig to set the appropriate output activation function (sigmoid for single-class, softmax for multi-class).
        For reconstruction tasks, you can use the `reconstruction_mode()` method to set the output activation to None.

        >>> cfg_segmentation = ReverseHalfUNetConfig().single_class_segmentation_mode()
        >>> model_segmentation = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_segmentation,
        >>>    )

        >>> cfg_reconstruction = ReverseHalfUNetConfig().reconstruction_mode()
        >>> model_reconstruction = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_reconstruction,
        >>>    )

        If deep supervision is desired, you can specify the levels at which to apply it using the `supervision_levels` argument.

        >>> model_deep_supervision = ReverseHalfUNet(
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
        >>> loaded_cfg = ReverseHalfUNetConfig.load("my_config.yaml")
        >>> model = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=loaded_cfg,
        >>>    )

    References:
        .. [1] H. Lu, Y. She, J. Tie, and S. Xu, Half-UNet: A Simplified ReverseHalfUNet Architecture for Medical Image Segmentation,
            Front. Neuroinformatics, vol. 16, Jun. 2022, doi: 10.3389/fninf.2022.911679.

        .. [2] O. Ronneberger, P. Fischer, and T. Brox, Reverse ReverseHalfUNet: Convolutional Networks for Biomedical Image Segmentation,
            May 18, 2015, arXiv: arXiv:1505.04597. doi: 10.48550/arXiv.1505.04597.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        cfg: ReverseHalfUNetConfig,
        supervision_levels: int | list[int] = 0,
    ):
        """ """
        super().__init__()

        self.hidden_channels = cfg.hidden_channels
        self.n_levels = cfg.n_levels

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

        # ---- pooling ----
        for i in range(self.n_levels - 1):
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

        # ---- deep supervision ----
        if isinstance(supervision_levels, int):
            supervision_levels = [supervision_levels]

        self.supervision_levels = supervision_levels

        self.fusion_type = cfg.fusion_type

        # ---- decoder ----
        self.decoders = torch.nn.ModuleList()
        self.out_blocks = torch.nn.ModuleList()

        for i in reversed(range(self.n_levels)):
            # in channels is hidden_channels if fusion_type is 'add' or if we are at the last level, otherwise it is hidden_channels * 2 (for concatenation)
            in_ch = (
                self.hidden_channels
                if self.fusion_type.strip().lower() == "add" or i == (self.n_levels - 1)
                else self.hidden_channels * 2
            )

            # upsample layer needs to know the number of channels in the input and output, as some upsampling layers (e.g., transposed conv) require them
            if i > 0:
                up = call_with_supported_kwargs(
                    get_image_layer(upsample_cfg[i - 1].name, rank),
                    {
                        "in_channels": self.hidden_channels,
                        "out_channels": self.hidden_channels,
                        **upsample_cfg[i - 1].kwargs,
                    },
                )
                self.ups.append(up)

            block = torch.nn.Sequential(
                *[
                    ImageConvBlock(
                        in_channels=in_ch if j == 0 else self.hidden_channels,
                        out_channels=self.hidden_channels,
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
                        in_channels=self.hidden_channels,
                        out_channels=out_channels,
                        rank=rank,
                        cfg=cfg.out_block_cfg,
                    )
                )

    def forward(self, x):
        """
        Forward pass through the Reverse ReverseHalfUNet.
        """
        x = self.stem(x)
        decoder_inputs = [x]

        # ---- pooling ----
        for pool in self.pools:
            x = pool(x)
            decoder_inputs.append(x)

        # ---- decoder ----
        decoder_outputs = []
        ctr = 0
        x = self.decoders[0](x)

        for i, (up, inp, dec) in enumerate(
            zip(self.ups, reversed(decoder_inputs[:-1]), self.decoders[1:], strict=True)
        ):
            if (self.n_levels - i - 1) in self.supervision_levels:
                decoder_outputs.append(self.out_blocks[ctr](x))
                ctr += 1

            x = up(x)

            # dynamic shape alignment
            if x.shape[2:] != inp.shape[2:]:
                inp = torch.nn.functional.interpolate(inp, size=x.shape[2:])

            x = apply_residual_connection(x, inp, connection_type=self.fusion_type)
            x = dec(x)

        if 0 in self.supervision_levels:
            decoder_outputs.append(self.out_blocks[ctr](x))

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

    model_deep_supervision = ReverseHalfUNet(
        rank=2,
        in_channels=32,
        out_channels=32,
        supervision_levels=[0, 1],  # Apply deep supervision at top 2 levels
    )
    model = model_deep_supervision
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print(model)

    x = torch.randn(1, 32, 64, 64)  # Example input for a 3D tensor

    def check_gradients(model, x):
        model.train()

        # Ensure input tracks gradients if you want to test input gradients
        x = x.requires_grad_(True)

        print(x.shape)
        # Forward
        y = model(x)
        if isinstance(y, list):
            for output in y:
                print(output.shape)
        else:
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
