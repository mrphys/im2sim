def get_model_config(name):
  """Get the default config for a specified type of model.

  Args:
    name: A `str`. The name of the model type.

  Returns:
    A config dictionary 

  Raises:
    ValueError: If the requested model config doesn't exist. 
  """
  try:
    return _CONFIGS[(name)]
  except KeyError as err:
    raise ValueError(
        f"Could not find config for model with name '{name}'") from err


_CONFIGS = {
    "Image2Flow":dict(cnn_filters=[16,48,96,192,384],
                cnn_kernel_size=3,
                cnn_res_depth=3,
                cnn_res_blocks_per_level=2,
                cnn_rank=3,
                cnn_norm_type="InstanceNorm",
                cnn_pool_type='MaxPool',
                cnn_pool_size=2,
                cnn_activation='leaky_relu',
                cnn_dropout_rate=0.3,
                projection_ids = [[3,4],[1,2],[0,1]],
                gnn_filters = [[384,288], [144,96], [64,32]],
                gnn_res_depth = 3,
                gnn_n_process_blocks = 1,
                gnn_n_deform_blocks = 3,
                template_edge_index=None,
                gnn_conv_type="ChebConv",
                gnn_conv_kwargs={'K':1},
                gnn_activation="leaky_relu",
                out_activation="leaky_relu",
                gnn_norm_type="InstanceNorm",
                batched_ops=False),

    "Image2Mesh":dict(cnn_filters=[16,32,64,128,256],
                cnn_kernel_size=3,
                cnn_res_depth=3,
                cnn_res_blocks_per_level=2,
                cnn_rank=3,
                cnn_norm_type="InstanceNorm",
                cnn_pool_type='MaxPool',
                cnn_pool_size=2,
                cnn_activation='leaky_relu',
                cnn_dropout_rate=0.3,
                projection_ids = [[3,4],[1,2],[0,1]],
                gnn_filters = [[384,288], [96, 64], [48,32]],
                gnn_res_depth = 3,
                gnn_n_process_blocks = 1,
                gnn_n_deform_blocks = 3,
                template_edge_index=None,
                gnn_conv_type="ChebConv",
                gnn_conv_kwargs={'K':1},
                gnn_activation="leaky_relu",
                out_activation="leaky_relu",
                gnn_norm_type="InstanceNorm",
                batched_ops=False)
    
}
