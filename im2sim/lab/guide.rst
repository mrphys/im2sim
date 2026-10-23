``lab`` is a set of controlled experiments for finding out what a GNN can and cannot do. Each
experiment answers one question on a graph where the right answer is known, so a failure points
at a property of the model rather than at the data.

======================  ================================================================
Question                Experiments
======================  ================================================================
How far do messages     ``jacobian_reach``, ``copy_task``, ``harmonic_extension``
travel?
How well does it        ``equivariance_error``, ``linear_probe``, ``resolution_invariance``,
understand geometry?    ``proximity_leakage``
How sharply can the     ``step_sweep``, ``spectral_gain``, ``boundary_layer``
signal change?
======================  ================================================================

*Untrained* experiments (``jacobian_reach``, ``equivariance_error``, ``linear_probe``,
``resolution_invariance``, ``proximity_leakage``, ``spectral_gain``) take a ready model and run in
seconds, so they work on a freshly initialised model, a checkpoint, or both. *Trained*
experiments (``copy_task``, ``harmonic_extension``, ``step_sweep``, ``boundary_layer``) fit a fresh
small model per case, so they take a **factory** instead.

Wrapping a model
----------------

:class:`~im2sim.lab.ModelAdapter` gives every model the same interface,
``adapter(graph, x=None) -> [N, C]``:

.. code-block:: python

    import torch_geometric.nn as gnn
    from im2sim.lab import ModelAdapter

    pyg_model = ModelAdapter(gnn.GCN(1, 32, num_layers=4, out_channels=2))   # model(x, edge_index)
    block = ModelAdapter(GraphConvBlock(1, 2, cfg))                          # model(graph) -> Data
    full = ModelAdapter(Im2SimGen2(...), output_key="cfd", image=my_image)   # model(image, graph)

The calling convention is detected from the model; pass ``kind=`` (``"pyg_xe"``, ``"pyg_data"``,
``"graph"``, ``"im2sim"``) or ``call_fn=`` to override it. Notes:

* Experiments inject their test signals into ``graph.<input_key>`` (``x`` by default).
* im2sim decoders with ``pred_feature_key="x"`` write into the *existing* ``x`` channels, so the
  output has as many channels as the input. For another output width predict into another key
  (e.g. ``"cfd"``) and set ``output_key``.
* ``hidden_states`` hooks ``MessagePassing`` layers by default; pass ``hidden_filter`` to choose
  others. ``Im2SimBase`` models need an ``image`` (or a rasterizer, in which case ``image=None``).

A trained experiment's factory is ``factory(in_channels, out_channels) -> ModelAdapter``:

.. code-block:: python

    factory = lambda i, o: ModelAdapter(gnn.GCN(i, 32, 4, out_channels=o))

Graphs
------

:mod:`im2sim.lab.graphs` builds toy graphs, all with ``pos`` and ``coords`` (the same tensor),
an undirected ``edge_index``, ``x`` and a ``boundary`` mask: ``path_graph``, ``ring_graph``,
``grid_graph``, ``tree_graph``, ``barbell_graph``, ``tube_graph`` (a vessel-like cylinder with
``wall_dist`` and an analytic ``poiseuille`` profile) and ``u_tube_graph`` (two limbs close in
space but far along the graph). ``resample`` rebuilds a graph at lower density.

A real graph is used through :func:`~im2sim.lab.graphs.from_mesh_graph`, which only needs
``coords`` (or ``pos``) and ``edge_index``:

.. code-block:: python

    from im2sim.lab import from_mesh_graph, jacobian_reach

    real = from_mesh_graph(torch.load("Mr_Test_2001.pt", weights_only=False))
    result = jacobian_reach(adapter, real)

Running and plotting
--------------------

Every experiment returns a :class:`~im2sim.lab.LabResult` with ``scalars`` (headline numbers) and
``arrays`` (curves and per-node data). :mod:`im2sim.lab.viz` plots them, and
:func:`~im2sim.lab.run_suite` runs a grid of models x experiments x seeds:

.. code-block:: python

    from functools import partial
    from im2sim.lab import run_suite, jacobian_reach, spectral_gain, viz, results_to_frame

    results = run_suite(
        models={"gcn4": factory, "gcn8": factory8},
        experiments={
            "reach": partial(jacobian_reach, graph=tube),
            "spectral": partial(spectral_gain, graph=tube),
        },
        seeds=(0, 1, 2),
    )
    viz.plot_reach_profile([r for r in results if r.name == "reach"])   # seeds -> mean/band
    results_to_frame(results)                                          # table of scalars

Reading the results
-------------------

* ``jacobian_reach``: ``receptive_field`` is the furthest hop with any influence. A non-zero
  ``global_coupling`` means something couples every node (a normalisation over the whole graph, a
  virtual node, global pooling), so the hop count alone understates the reach. ``DefaultGraphNorm``
  does this.
* ``equivariance_error``: coordinates only matter if they reach the model. Pass
  ``input_fn=lambda g: g.pos`` to test a model that is fed coordinates as features.
* ``linear_probe``: compare each layer with the ``input`` row; the gain is what the network added.
* ``spectral_gain``: the Dirichlet ratio is exactly 1 for linear filters that share the Laplacian
  eigenbasis; deviations show nonlinear or position-dependent frequency mixing.
* Trained experiments fit a single graph (``step_sweep``, ``boundary_layer``) or a small family
  (``copy_task``, ``harmonic_extension``). They measure what a model *can represent* at this
  size and training budget, so compare models under the same ``steps``/``lr``.
