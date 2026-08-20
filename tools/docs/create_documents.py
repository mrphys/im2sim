# Copyright 2026 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RST document generator."""

import dataclasses
import inspect
import os
import string
import sys
import typing
import ast
import importlib



DOCS_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(DOCS_PATH, '..', '..')
TEMPLATES_PATH = os.path.join(DOCS_PATH, 'templates')
API_DOCS_PATH = os.path.join(DOCS_PATH, 'api_docs')

sys.path.insert(0, ROOT_PATH)

from im2sim.utils import api_util

# Create API docs directory.
os.makedirs(os.path.join(API_DOCS_PATH, 'im2sim'), exist_ok=True)

# Read the index template.
with open(os.path.join(TEMPLATES_PATH, 'index.rst'), 'r') as f:
  INDEX_TEMPLATE = string.Template(f.read())

im2sim_DOC_TEMPLATE = string.Template(
"""
im2sim
=========

.. automodule:: im2sim

Modules
-------

.. autosummary::
    :nosignatures:

    ${namespaces}

""")

MODULE_DOC_TEMPLATE = string.Template(
"""im2sim.${module}
=======${underline}

.. automodule:: im2sim.${module}

Classes
-------

.. autosummary::
    :toctree: ${module}
    :template: ${module}/class.rst
    :nosignatures:

    ${classes}

Functions
---------

.. autosummary::
    :toctree: ${module}
    :template: ${module}/function.rst
    :nosignatures:

    ${functions}
""")


@dataclasses.dataclass
class Module:
  """A module."""
  classes: typing.List[str] = dataclasses.field(default_factory=list)
  functions: typing.List[str] = dataclasses.field(default_factory=list)


# def get_public_symbols_from_init(init_path):
#     """Get public classes and functions exported by an __init__.py file."""
#     with open(init_path, "r") as f:
#         tree = ast.parse(f.read(), filename=str(init_path))

#     classes = []
#     functions = []

#     for node in tree.body:
#         if isinstance(node, ast.ImportFrom):
#             for alias in node.names:
#                 if alias.name == "*":
#                     continue

#                 # We only want names exposed by the __init__.py.
#                 name = alias.asname or alias.name
#                 classes.append(name)

#         elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
#             if not node.name.startswith("_"):
#                 if isinstance(node, ast.ClassDef):
#                     classes.append(node.name)
#                 else:
#                     print(node.name)
#                     functions.append(node.name)

#     return classes, functions


def get_public_symbols_from_init(module_name):
    module = importlib.import_module(module_name)

    classes = []
    functions = []

    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue

        if inspect.isclass(obj):
            classes.append(name)

        elif inspect.isfunction(obj):
            functions.append(name)

    return classes, functions


code_path = os.path.join(ROOT_PATH, "im2sim")


namespaces = [
    name
    for name in os.listdir(code_path)
    if os.path.isdir(os.path.join(code_path, name))
]

modules = {}
for namespace in namespaces:
    classes, functions =  get_public_symbols_from_init("im2sim."+namespace)
    modules[namespace] = Module(classes=classes, functions=functions)

# modules = {namespace: Module() for namespace in api_util.get_submodule_names()}



# for name, symbol in api_util.get_api_symbols().items():
#   name = api_util.get_canonical_name_for_symbol(symbol)
#   namespace, name = name.split('.', maxsplit=1)

#   if inspect.isclass(symbol):
#     modules[namespace].classes.append(name)
#   elif inspect.isfunction(symbol):
#     modules[namespace].functions.append(name)

# Write namespace templates.
for name, module in modules.items():
  classes = '\n    '.join(sorted(set(module.classes)))
  functions = '\n    '.join(sorted(set(module.functions)))

  filename = os.path.join(API_DOCS_PATH, f'im2sim/{name}.rst')
  with open(filename, 'w') as f:
    f.write(MODULE_DOC_TEMPLATE.substitute(
        module=name,
        underline='=' * len(name),
        classes=classes,
        functions=functions))

# Write top-level API doc im2sim.rst.
filename = os.path.join(API_DOCS_PATH, 'im2sim.rst')
with open(filename, 'w') as f:
  # namespaces = api_util.get_submodule_names()
  namespaces = list(modules.keys())
  f.write(im2sim_DOC_TEMPLATE.substitute(
      namespaces='\n    '.join(sorted(namespaces))))

# Write index.rst.
filename = os.path.join(DOCS_PATH, 'index.rst')
with open(filename, 'w') as f:
  # namespaces = api_util.get_submodule_names()
  namespaces = list(modules.keys())
  namespaces = ['api_docs/im2sim/' + namespace for namespace in namespaces]
  f.write(INDEX_TEMPLATE.substitute(
      namespaces='\n   '.join(sorted(namespaces))))
