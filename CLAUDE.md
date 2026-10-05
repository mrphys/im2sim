# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- **Run all tests**: `pytest` or `pytest tests/`
- **Run unit tests only**: `pytest tests/unit/`
- **Run integration tests only**: `pytest tests/integration/`
- **Run specific test file**: `pytest tests/unit/layers/test_image_blocks.py`
- **Run single test**: `pytest tests/unit/layers/test_image_blocks.py::test_name`
- **Run with coverage**: `pytest --cov=im2sim tests/`
- **Run with verbose output**: `pytest -v tests/`

### Linting & Formatting
- **Check linting**: `ruff check im2sim/ tests/`
- **Fix linting and formatting issues**: `ruff check --fix im2sim/ tests/ && ruff format im2sim/ tests/`
- **Format only**: `ruff format im2sim/ tests/`

### Pre-commit
- **Run pre-commit on all files**: `pre-commit run --all-files`
- **Install pre-commit hooks**: `pre-commit install`

### Installation & Setup
- **Install in development mode**: `pip install -e ".[dev]"`
- **Install with mesh support**: `pip install -e ".[dev,mesh]"`
- **Install PyTorch Geometric addons** (after PyTorch/CUDA versions are installed): `im2sim-install-pyg-addons`

## Architecture Overview

### Core Components

The library is organized around **four main architectural layers**:

1. **Models** (`im2sim/models/`): High-level PyTorch models combining image and graph processing
   - `unet.py`, `halfunet.py`, `reverse_halfunet.py`: Image encoders/decoders
   - `graph_decoders.py`: GNN-based graph processors (PyTorch Geometric)
   - `image_to_graph.py`: Hybrid models that bridge image and graph domains
   - `im2sim_models.py`: Full end-to-end models combining projections, rasterization, and decoders

2. **Layers** (`im2sim/layers/`): Reusable PyTorch modules and building blocks
   - `image_blocks.py`: Conv blocks (2D/3D), normalization, dropout, attention
   - `graph_blocks.py`: Graph processing blocks (message passing, aggregation)
   - `custom_image_layers.py`, `custom_graph_layers.py`: Custom layer implementations
   - `projections.py`: Feature projection from image to graph space (e.g., TrilinearProjection)
   - `rasterization.py`: Convert graph outputs back to image space (MaskRasterizer)

3. **Data** (`im2sim/data/`): Data pipeline and preprocessing
   - `core.py`: Core data abstractions
   - `pca.py`: Dimensionality reduction
   - `scaling.py`: Normalization and scaling
   - Mesh utilities for VTK/PyVista integration

4. **Configs** (`im2sim/configs/`): Configuration classes for reproducibility
   - Each major component has a corresponding config class (LayerConfig, ModelConfig, etc.)
   - Enables composition of complex models via nested configs

### Supporting Modules

- **Losses** (`im2sim/losses/`): Custom loss functions (SSIM, feature-based, mesh-based)
- **Transforms** (`im2sim/transforms/`): Data transformations and augmentations
- **Utils** (`im2sim/utils/`): Helper functions for layer creation, activation selection, etc.
- **Mesh Ops** (`im2sim/mesh_ops/`): Mesh processing operations (PyVista/VTK)
- **Plot** (`im2sim/plot/`): Visualization utilities

### Design Patterns

- **Signature Checking**: Complex models (e.g., in `im2sim_models.py`) validate component interfaces at initialization to catch configuration errors early
- **Config-Driven Architecture**: Nearly all components accept configuration objects to ensure reproducibility and enable dynamic model composition
- **Modular Blocks**: Image and graph blocks are designed to be composable into larger models

## Code Conventions

- **Line length**: 100 characters (enforced by ruff)
- **Python version**: 3.10+
- **Import sorting**: Handled automatically by ruff
- **Quote style**: Double quotes
- **Formatting**: Automatic via ruff format

## Testing Strategy

- **Unit tests** (`tests/unit/`): Test individual layers, blocks, and utilities in isolation
- **Integration tests** (`tests/integration/`): Test full model pipelines and data flow
- **Key patterns**:
  - Use pytest parametrization for testing multiple configurations
  - Integration tests often test model instantiation and forward passes with synthetic data
  - Hypothesis library is available for property-based testing

## Key Dependencies

- **torch** (≥2.3): Core deep learning framework
- **torch_geometric**: Graph neural networks and PyTorch Geometric operations
- **numpy**: Numerical computing
- **pyvista** (optional, for mesh support): 3D data visualization and mesh operations
- **scikit-learn**: Additional ML utilities (preprocessing, etc.)

## Common Workflows

### Adding a new layer/block
1. Create module in `im2sim/layers/`
2. Add corresponding config in `im2sim/configs/`
3. Add unit test in `tests/unit/layers/`
4. Ensure it follows the expected forward signature for composition

### Adding a new model
1. Create model in `im2sim/models/`
2. Add config in `im2sim/configs/`
3. Add integration test in `tests/integration/models/`
4. Document expected input/output shapes

### Adding tests
1. Unit tests go in `tests/unit/` mirroring package structure
2. Integration tests go in `tests/integration/`
3. Run `pytest --cov=im2sim` to check coverage
4. Mark slow/GPU tests with `@pytest.mark.slow` or similar as appropriate
