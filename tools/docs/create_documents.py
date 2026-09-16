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

${module_guide_text}

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

# Write namespace templates.
for name, module in modules.items():
  classes = '\n    '.join(sorted(set(module.classes)))
  functions = '\n    '.join(sorted(set(module.functions)))

  filename = os.path.join(API_DOCS_PATH, f'im2sim/{name}.rst')

  with open(f"{ROOT_PATH}/im2sim/{name}/guide.rst", "r") as src:
    module_guide_text = src.read()

  with open(filename, 'a') as f:
    f.write(MODULE_DOC_TEMPLATE.substitute(
        module=name,
        underline='=' * len(name),
        module_guide_text = module_guide_text,
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