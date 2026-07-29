import pytest
import torch
import torch.nn as nn

from im2sim.layers import (
    ImageConvBlock,
    ImageConvResBlock,
    ImageEncoder,
    ImageResEncoder,
    ImageDecoder,
)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def image_2d():
    # BCHW
    return torch.randn(2, 3, 64, 64, requires_grad=True)


@pytest.fixture
def image_3d():
    # BCDHW
    return torch.randn(2, 3, 32, 32, 32, requires_grad=True)


@pytest.fixture
def small_filters():
    return (8, 16, 32)


# ---------------------------------------------------------
# ImageConvBlock
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "rank,input_shape,expected_spatial",
    [
        (2, (2, 3, 64, 64), (64, 64)),
        (3, (2, 3, 32, 32, 32), (32, 32, 32)),
    ],
)
def test_image_conv_block_shapes(rank, input_shape, expected_spatial):

    model = ImageConvBlock(
        in_channels=3,
        filters=16,
        depth=2,
        rank=rank,
    )

    x = torch.randn(*input_shape)

    y = model(x)

    assert y.shape == (
        input_shape[0],
        16,
        *expected_spatial,
    )


@pytest.mark.parametrize(
    "activation",
    [
        "ReLU",
        "relu",
        "gelu",
        "sigmoid",
        None,
    ],
)
def test_image_conv_block_activations(activation):

    model = ImageConvBlock(
        in_channels=3,
        filters=8,
        rank=2,
        activation=activation,
    )

    x = torch.randn(1, 3, 32, 32)

    y = model(x)

    assert y.shape == (1, 8, 32, 32)



@pytest.mark.parametrize(
    "norm",
    [
        None,
        "BatchNorm",
        "InstanceNorm",
    ],
)
def test_image_conv_block_normalisation(norm):

    model = ImageConvBlock(
        in_channels=3,
        filters=8,
        rank=2,
        norm_type=norm,
    )

    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 8, 32, 32)



def test_image_conv_block_dropout():

    model = ImageConvBlock(
        in_channels=3,
        filters=8,
        rank=2,
        dropout_rate=0.5,
    )

    assert isinstance(model.drop, nn.Dropout2d)

    x = torch.randn(2,3,32,32)

    model.train()

    y1 = model(x)
    y2 = model(x)

    assert not torch.equal(y1, y2)



def test_image_conv_block_gradients():

    model = ImageConvBlock(
        in_channels=3,
        filters=8,
        rank=2,
        depth=3,
    )

    x = torch.randn(
        2,
        3,
        32,
        32,
        requires_grad=True,
    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    for p in model.parameters():
        assert p.grad is not None



def test_image_conv_block_parameterisation():

    model = ImageConvBlock(
        in_channels=3,
        filters=16,
        depth=4,
        kernel_size=5,
        rank=2,
    )

    # 4 convolution layers
    assert len(model.convs) == 4

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    assert params > 0



# ---------------------------------------------------------
# Residual Block
# ---------------------------------------------------------

def test_res_block_shape():

    model = ImageConvResBlock(
        in_channels=3,
        filters=16,
        rank=2,
    )

    x = torch.randn(
        2,
        3,
        64,
        64,
    )

    y = model(x)

    assert y.shape == (
        2,
        16,
        64,
        64,
    )



def test_res_block_gradient():

    model = ImageConvResBlock(
        in_channels=3,
        filters=8,
        rank=2,
    )

    x = torch.randn(
        1,
        3,
        32,
        32,
        requires_grad=True,
    )

    y = model(x)

    y.mean().backward()

    assert x.grad is not None



def test_res_block_depth_parameter():

    model = ImageConvResBlock(
        in_channels=3,
        filters=8,
        depth=5,
    )

    # main block receives depth-2 layers
    assert len(model.main_conv.convs) == 3



# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "rank",
    [2,3],
)
def test_image_encoder_shapes(rank, small_filters):

    model = ImageEncoder(
        in_channels=3,
        filters=small_filters,
        rank=rank,
    )


    if rank == 2:
        x = torch.randn(
            1,
            3,
            64,
            64
        )

        expected = [
            (1,8,64,64),
            (1,16,32,32),
            (1,32,16,16),
        ]

    else:
        x = torch.randn(
            1,
            3,
            32,
            32,
            32,
        )

        expected = [
            (1,8,32,32,32),
            (1,16,16,16,16),
            (1,32,8,8,8),
        ]


    outputs = model(x)

    assert len(outputs)==len(expected)

    for out, shape in zip(outputs, expected):
        assert out.shape == shape



def test_encoder_parameters():

    model = ImageEncoder(
        in_channels=3,
        filters=(8,16),
        rank=2,
        conv_blocks_per_level=2,
    )


    assert len(model.conv_blocks)==2

    params=sum(
        p.numel()
        for p in model.parameters()
    )

    assert params > 0



# ---------------------------------------------------------
# Residual Encoder
# ---------------------------------------------------------

def test_res_encoder_outputs():

    model = ImageResEncoder(
        in_channels=3,
        filters=(8,16,32),
        rank=2,
        res_blocks_per_level=2,
    )

    x=torch.randn(
        1,
        3,
        64,
        64,
    )

    outputs=model(x)

    assert len(outputs)==3

    assert outputs[0].shape == (
        1,
        8,
        64,
        64
    )

    assert outputs[-1].shape == (
        1,
        32,
        16,
        16
    )



# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------

def test_decoder_reconstructs_resolution():

    encoder = ImageEncoder(
        in_channels=3,
        filters=(8,16,32),
        rank=2,
    )

    decoder = ImageDecoder(
        filters=(8,16,32),
        rank=2,
    )


    x=torch.randn(
        1,
        3,
        64,
        64,
    )


    enc_features=encoder(x)

    out=decoder(enc_features)


    assert out.shape == (
        1,
        8,
        64,
        64,
    )



@pytest.mark.parametrize(
    "skip",
    [
        True,
        False,
    ],
)
def test_decoder_skip_parameter(skip):

    decoder = ImageDecoder(
        filters=(8,16,32),
        rank=2,
        skip=skip,
    )

    assert decoder.skip == skip



@pytest.mark.parametrize(
    "upsample_type",
    [
        "Upsample",
        "ConvTranspose",
    ],
)
def test_decoder_upsampling_modes(upsample_type):

    encoder = ImageEncoder(
        in_channels=3,
        filters=(8,16,32),
        rank=2,
    )

    decoder = ImageDecoder(
        filters=(8,16,32),
        rank=2,
        upsample_type=upsample_type,
    )


    x = torch.randn(
        1,
        3,
        64,
        64,
    )


    features = encoder(x)

    out = decoder(features)


    assert out.shape == (
        1,
        8,
        64,
        64,
    )



# ---------------------------------------------------------
# End-to-end gradient test
# ---------------------------------------------------------

def test_encoder_decoder_end_to_end_gradient():

    encoder = ImageEncoder(
        in_channels=3,
        filters=(8,16),
        rank=2,
    )

    decoder = ImageDecoder(
        filters=(8,16),
        rank=2,
    )


    x=torch.randn(
        2,
        3,
        32,
        32,
        requires_grad=True,
    )


    output=decoder(
        encoder(x)
    )


    loss=output.mean()

    loss.backward()


    assert x.grad is not None

    encoder_grads=[
        p.grad
        for p in encoder.parameters()
        if p.requires_grad
    ]

    decoder_grads=[
        p.grad
        for p in decoder.parameters()
        if p.requires_grad
    ]


    assert all(
        g is not None
        for g in encoder_grads
    )

    assert all(
        g is not None
        for g in decoder_grads
    )

# ---------------------------------------------------------
# Decoder crop logic
# ---------------------------------------------------------

def test_decoder_match_size_crops_skip_connection():

    decoder = ImageDecoder(
        filters=(8, 16),
        rank=2,
        skip=True,
    )

    # decoder feature map
    x = torch.randn(
        1,
        16,
        30,
        30,
    )

    # encoder skip feature map is larger
    skip = torch.randn(
        1,
        8,
        32,
        32,
    )

    cropped = decoder._match_size(x, skip)

    assert cropped.shape == (
        1,
        8,
        30,
        30,
    )


def test_decoder_match_size_center_crop():

    decoder = ImageDecoder(
        filters=(8,16),
        rank=2,
    )

    # Put a known pattern in the skip tensor
    skip = torch.arange(
        32 * 32
    ).reshape(
        1,
        1,
        32,
        32,
    ).float()


    x = torch.zeros(
        1,
        16,
        28,
        28,
    )


    cropped = decoder._match_size(
        x,
        skip,
    )


    # difference is 4 pixels -> crop 2 from each side
    expected = skip[
        ...,
        2:30,
        2:30,
    ]


    assert torch.equal(
        cropped,
        expected,
    )



def test_decoder_match_size_no_crop():

    decoder = ImageDecoder(
        filters=(8,16),
        rank=2,
    )

    x = torch.randn(
        1,
        16,
        32,
        32,
    )

    skip = torch.randn(
        1,
        8,
        32,
        32,
    )


    output = decoder._match_size(
        x,
        skip,
    )

    # should return same tensor
    assert output.shape == skip.shape
    assert torch.equal(output, skip)



# ---------------------------------------------------------
# Full decoder with odd image dimensions
# ---------------------------------------------------------

def test_decoder_handles_odd_spatial_dimensions():

    encoder = ImageEncoder(
        in_channels=3,
        filters=(8,16,32),
        rank=2,
    )

    decoder = ImageDecoder(
        filters=(8,16,32),
        rank=2,
        skip=True,
    )


    # Odd dimensions trigger possible mismatch
    x = torch.randn(
        1,
        3,
        65,
        65,
    )


    features = encoder(x)

    output = decoder(features)


    assert output.shape[2:] == (
        64,
        64,
    )