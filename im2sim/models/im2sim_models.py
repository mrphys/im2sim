import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"

import torch
import inspect
import torch_geometric as pyg


import sys
sys.path.append("/Users/anirudh/Documents/im2sim/im2sim")  # Add the parent directory to the Python path



from im2sim.models.graph_decoders import SimpleGraphDecoder
from im2sim.configs.graph_blocks import GraphConvBlockConfig
from im2sim.configs.halfunet import HalfUNetConfig
from im2sim.models.halfunet import HalfUNet
from im2sim.configs.core import LayerConfig
from im2sim.configs.graph_decoder import SimpleGraphDecoderConfig

from im2sim.layers.projections import TrilinearProjection
from im2sim.layers.rasterization import MaskRasterizer

from im2sim.utils.layer_util import get_image_layer



# TODO: Make an iteration config that allows rasterisation and feedback to make the iteration functional. 

def check_graph_decoder_signature(graph_decoder: torch.nn.Module):
    if list(inspect.signature(graph_decoder.forward).parameters.keys()) != ['in_graph', 'projected_features']:
        raise ValueError("Graph decoder must have a forward method with signature (graph, projected_features)")
    
def check_projection_signature(projection: torch.nn.Module):
    if list(inspect.signature(projection.forward).parameters.keys()) != ['image_features', 'graph']:
        raise ValueError("Projection must have a forward method with signature (image_features, coords)")

def check_rasterizer_signature(rasterizer: torch.nn.Module):
    if list(inspect.signature(rasterizer.forward).parameters.keys()) != ['graph', 'image_input']:
        raise ValueError("Rasterizer must have a forward method with signature (graph, image_input)")


class Im2SimBase(torch.nn.Module):
    """
    A base class for the Im2Sim model that combines an image encoder, a graph decoder, and optional rasterization.

    Args:
        image_shape (tuple[int, int, int]): The shape of the input image.
        image_encoder (torch.nn.Module): The image encoder module.
        graph_decoder (torch.nn.Module): The graph decoder module.
        projections (torch.nn.Module | list[torch.nn.Module]): The projection module(s) that project image features onto the graph.
        rasterizer (torch.nn.Module, optional): The rasterizer module that generates an image from the graph. Default is None.
        n_iters (int, optional): The number of iterations to run the model. Default is 1.
        return_intermediate_graphs (bool, optional): Whether to return intermediate graphs after each iteration. Default is False.
    """

    def __init__(self,
                 image_shape: tuple[int, int, int],
                 image_encoder: torch.nn.Module,
                 graph_decoder: torch.nn.Module,
                 projections: torch.nn.Module | list[torch.nn.Module], 
                 rasterizer: torch.nn.Module = None,
                 n_iters: int = 1, 
                 return_intermediate_graphs: bool = False
                 ):
        super().__init__()
        self.image_shape = image_shape
        self.image_encoder = image_encoder

        check_graph_decoder_signature(graph_decoder)
        
        self.graph_decoder = graph_decoder

        if isinstance(projections, list):
            for proj in projections:
                check_projection_signature(proj)
        else:
            check_projection_signature(projections)

        self.projections = projections

        if rasterizer is not None:
            check_rasterizer_signature(rasterizer)

        self.rasterizer = rasterizer
        self.n_iters = n_iters
        self.return_intermediate_graphs = return_intermediate_graphs

    def forward(self, image_input: torch.Tensor, in_graph: pyg.data.Data) -> pyg.data.Data | list[pyg.data.Data]:
        """
        Forward pass of the Im2Sim model.

        Args:
            image_input (torch.Tensor): The input image tensor. If None, the rasterizer will be used to generate an image from the graph.
            in_graph (pyg.data.Data): The input graph data.
        """
        graph = in_graph.clone()
        # If image_input is None, use the rasteriser to generate an image from the graph
        if image_input is None:
            if self.rasterizer is None:
                raise ValueError("Image input is None and no rasteriser is provided.")
            image_input = torch.zeros(1, self.image_encoder.in_channels, *self.image_shape, device=graph.x.device)
            image_input = self.rasterizer(graph, image_input)

        if 'coords' not in graph:
            raise ValueError("Graph must have 'coords' attribute for projection.")

        # Encode the image features only once if rasteriser is not provided
        if self.rasterizer is None:
            image_features = self.image_encoder(image_input)

        out_graphs = []
        for _ in range(self.n_iters):

            # Encode the image each iteration if rasteriser is provided
            if self.rasterizer is not None:
                image_features = self.image_encoder(image_input)
            
            if not isinstance(self.projections, list):
                projected_features = self.projections(image_features, graph)
            else:
                projected_features = []
                for feat, proj in zip(image_features, self.projections):
                    projected_features.append(proj(feat, graph))

            # Decode the graph
            graph = self.graph_decoder(graph, projected_features)
            out_graphs.append(graph)

            # Rasterise if rasteriser is provided
            if self.rasterizer is not None:
                image_input = self.rasterizer(graph, image_input)
            
        if self.return_intermediate_graphs:
            return out_graphs
        
        return graph
    

class Im2SimGen2(Im2SimBase):
    """
    A specific implementation of the Im2Sim model that uses a HalfUNet for image encoding and a SimpleGraphDecoder for graph decoding.

    Args:
        image_shape (tuple[int, int, int]): The shape of the input image.
        image_channels (int): The number of channels in the input image.
        projection_channels (int): The number of channels in the projected image features.
        graph_channels (int): The number of channels in the input graph features.
        out_channels (int): The number of channels in the output graph features.
        encoder_cfg (HalfUNetConfig): Configuration for the HalfUNet image encoder.
        decoder_cfg (SimpleGraphDecoderConfig): Configuration for the SimpleGraphDecoder graph decoder.
        projection (torch.nn.Module): The projection module that projects image features onto the graph.
        rasterizer (torch.nn.Module, optional): The rasterizer module that generates an image
    """

    def __init__(self,
                 image_shape: tuple[int, int, int],
                 image_channels: int,
                 projection_channels: int,
                 graph_channels: int,
                 out_channels: int,
                 encoder_cfg: HalfUNetConfig,
                 decoder_cfg: SimpleGraphDecoderConfig,
                 projection: torch.nn.Module, 
                 rasterizer: torch.nn.Module = None,
                 n_iters: int = 1,
                 return_intermediate_graphs: bool = False):
        
        image_encoder = HalfUNet(in_channels=image_channels, 
                                 out_channels=projection_channels, 
                                 rank=3, 
                                 cfg=encoder_cfg)   
        
        in_graph_channels = graph_channels + projection_channels
        if decoder_cfg.pred_feature_key != 'x':
            if decoder_cfg.pred_feature_channels is not None:
                in_graph_channels += len(decoder_cfg.pred_feature_channels)
            else:
                in_graph_channels += out_channels
    
        graph_decoder = SimpleGraphDecoder(in_channels=in_graph_channels, 
                                           out_channels=out_channels,
                                           cfg=decoder_cfg)
        # projection = get_image_layer(projection_cfg.name, rank=0)(**projection_cfg.kwargs)
        # rasterizer = get_image_layer(rasterizer_cfg.name, rank=0)(**rasterizer_cfg.kwargs) if rasterizer_cfg is not None else None

        super().__init__(image_shape=image_shape,
                        image_encoder=image_encoder,
                        graph_decoder=graph_decoder,
                        projections=projection,
                        rasterizer=rasterizer,
                        n_iters=n_iters,
                        return_intermediate_graphs=return_intermediate_graphs)


if __name__ == "__main__":

    device = torch.device("cpu")
    # Example usage
    encoder_cfg = HalfUNetConfig()
    # block_cfg = GraphConvBlockConfig()
    decoder_cfg = SimpleGraphDecoderConfig(pred_feature_key='coords')
    cfd_cfg = SimpleGraphDecoderConfig(pred_feature_key='cfd')
    projection = TrilinearProjection(128)
    rasterizer = MaskRasterizer()

    coords_model = Im2SimGen2(
        image_shape=(128, 128, 128),
        image_channels=3,
        projection_channels=64,
        graph_channels=32,
        out_channels=3,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        projection=projection,
        rasterizer=rasterizer,
        n_iters=2,
        return_intermediate_graphs=False
    ).to(device)

    cfd_model = Im2SimGen2(
        image_shape=(128, 128, 128),
        image_channels=1,
        projection_channels=64,
        graph_channels=32,
        out_channels=4,
        encoder_cfg=encoder_cfg,
        decoder_cfg=cfd_cfg,
        projection=projection,
        rasterizer=rasterizer,
        n_iters=1,
        return_intermediate_graphs=False
    ).to(device)

    in_graph = pyg.data.Data(x=torch.randn(10, 32), 
                             batch = torch.zeros(10, dtype=torch.long),
                             coords=torch.rand(10, 3), 
                             edge_index=torch.tensor([[0, 1], [1, 2]])).to(device)
    in_graph.x = in_graph.x.requires_grad_(True)

    image_input = torch.randn(1, 3, 128, 128, 128).to(device)  # Example image input
    image_input = image_input.requires_grad_(True)
    image_copy = image_input.clone()

    coords_graph = coords_model(image_input, in_graph)
    out_graph = cfd_model(None, coords_graph)
    print(out_graph.cfd)
    print(in_graph.coords - out_graph.coords)
    loss = out_graph.x.sum()  # Example loss
    loss.backward()
    assert out_graph.x.shape[-1] == 32, "Output graph channels do not match expected output channels"
    assert image_input.grad is not None, "Image input gradient is None"
    assert torch.isfinite(image_input.grad).all(), "Image input gradient contains NaN/Inf"
    assert in_graph.x.grad is not None, "Input graph gradient is None"
    assert torch.isfinite(in_graph.x.grad).all(), "Input graph gradient contains NaN/Inf"

