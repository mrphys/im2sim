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
"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from os import path
import inspect
import operator
import packaging.version
import re
import sys
import types

import conf_helper



# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, path.abspath('../..'))


# -- Project information -----------------------------------------------------

ROOT = path.abspath(path.join(path.dirname(__file__), '../..'))

ABOUT = {}
with open(path.join(ROOT, "im2sim/__about__.py")) as f:
  exec(f.read(), ABOUT)
_version = packaging.version.Version(ABOUT['__version__'])

project = ABOUT['__title__']
copyright = ABOUT['__copyright__']
author = ABOUT['__author__']
release = _version.public
version = '.'.join(map(str, (_version.major, _version.minor)))



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.autosummary',
  "sphinx.ext.intersphinx",
  'sphinx.ext.linkcode',
  'sphinx.ext.autosectionlabel',
  'myst_nb',
  'sphinx_sitemap'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autosectionlabel_prefix_document = True

autodoc_typehints = "description"

# Make Sphinx resolve types automatically
python_use_unqualified_type_names = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# Add the reference to the bibliography file.

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "templates"]

# Do not add full qualification to objects' signatures.
add_module_names = False

# For classes, list the documentation of both the class and the `__init__`
# method.
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------

html_title = 'IM2SIM Documentation'
html_logo = '../assets/im2sim_logo.png'
html_favicon = '../assets/im2sim_logo.png'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../assets']

# https://sphinx-book-theme.readthedocs.io/en/latest/tutorials/get-started.html
html_theme_options = {
    'repository_url': 'https://github.com/mrphys/im2sim',
    'use_repository_button': True,
    'launch_buttons': {
        'colab_url': "https://colab.research.google.com/"
    },
    'path_to_docs': 'docs'
}

html_css_files = [
    'https://fonts.googleapis.com/css?family=Roboto|Roboto+Mono',
]

# Additional files to copy to output directory.
html_extra_path = ['robots.txt']

# For sitemap generation.
html_baseurl = 'https://mrphys.github.io/im2sim/'
sitemap_url_scheme = '{link}'

# For autosummary generation.
autosummary_filename_map = conf_helper.AutosummaryFilenameMap()

# -- Options for MyST ----------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/authoring/jupyter-notebooks.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# https://myst-nb.readthedocs.io/en/latest/authoring/basics.html
source_suffix = [
    '.rst',
    '.md',
    '.ipynb'
]

# Do not execute notebooks.
# https://myst-nb.readthedocs.io/en/latest/computation/execute.html
nb_execution_mode = "off"


import im2sim


def linkcode_resolve(domain, info):
  """Find the GitHub URL where an object is defined.

  Args:
    domain: The language domain. This is always `py`.
    info: A `dict` with keys `module` and `fullname`.

  Returns:
    The GitHub URL to the object, or `None` if not relevant.
  """

  if domain != 'py':
    return None

  # Obtain fully-qualified name of object.
  qualname = info['module'] + '.' + info['fullname']
  print(qualname)

  # Remove the `im2sim` bit.
  qualname = qualname.split('.', maxsplit=1)[-1]
  print(qualname)

  # Get the object.
  try:
    obj = operator.attrgetter(qualname)(im2sim)
  except AttributeError:
    return None

  # We only add links to classes and functions.
  if not isinstance(obj, (type, types.FunctionType)):
    return None

  # Get the file name of the current object.
  file = inspect.getsourcefile(obj)

  # If no file, we're done. This happens for C++ ops.
  if file is None:
    return None

  # When using TF's deprecation decorators, `getsourcefile` returns the
  # `deprecation.py` file where the decorators are defined instead of the
  # file where the object is defined.
  if 'deprecation' in file:
    return None

  # Crop anything before `im2sim`.
  if 'im2sim' not in file:
    return None

  index = file.index('im2sim')
  file = file[index:]

  # Base URL.
  url = 'https://github.com/mrphys/im2sim'

  # Add version blob.
  url += '/blob/main'

  # Add file.
  url += '/' + file

  # Try to add line numbers.
  try:
    lines, start = inspect.getsourcelines(obj)
    stop = start + len(lines) - 1
  except OSError:
    return url

  # Add line numbers.
  url += '#L' + str(start) + '-L' + str(stop)

  return url


# def linkcode_resolve(domain, info):
#   """Find the GitHub URL where an object is defined.

#   Args:
#     domain: The language domain. This is always `py`.
#     info: A `dict` with keys `module` and `fullname`.

#   Returns:
#     The GitHub URL to the object, or `None` if not relevant.
#   """

#   # Obtain fully-qualified name of object.
#   qualname = info['module'] + '.' + info['fullname']
#   print(qualname)
#   # Remove the `im2sim` bit.
#   qualname = qualname.split('.', maxsplit=1)[-1]
#   print(qualname)

#   # Get the object.
#   # obj = operator.attrgetter(qualname)(im2sim)
#   try:
#     obj = operator.attrgetter(qualname)(im2sim)
#   except AttributeError:
#       return None
#   # We only add links to classes (type `type`) and functions
#   # (type `types.FunctionType`).
#   if not isinstance(obj, (type, types.FunctionType)):
#     return None

#   # Get the file name of the current object.
#   file = inspect.getsourcefile(obj)
#   # If no file, we're done. This happens for C++ ops.
#   if file is None:
#     return None
#   # When using TF's deprecation decorators, `getsourcefile` returns the
#   # `deprecation.py` file where the decorators are defined instead of the
#   # file where the object is defined. This should probably be fixed on the
#   # decorators themselves. For now, we just don't add the link for deprecated
#   # objects.
#   if 'deprecation' in file:
#     return None
#   # Crop anything before `im2sim\src`. This path is system
#   # dependent and we don't care about it.
#   index = file.index('im2sim')
#   file = file[index:]

#   # Base URL.
#   url = 'https://github.com/mrphys/im2sim'
#   # Add version blob.
#   url += '/blob/main'
#   # Add file.
#   url += '/' + file

#   # Try to add line numbers. This will not work when the class is defined
#   # dynamically. In that case we point to the file, but no line number.
#   try:
#     lines, start = inspect.getsourcelines(obj)
#     stop = start + len(lines) - 1
#   except OSError:
#     # Could not get source lines.
#     return url

#   # Add line numbers.
#   url += '#L' + str(start) + '-L' + str(stop)

#   return url


# -- Hyperlinks --------------------------------------------------------------
# Common types and constants in the API docs are enriched with hyperlinks to
# their corresponding docs.

# The following dictionary specifies type names and the corresponding links.
# The link is only added if the name has inline code format, e.g. ``foo``.
COMMON_TYPES_LINKS = {
    # Python standard types.
    'int': 'https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex',
    'float': 'https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex',
    'complex': 'https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex',
    'str': 'https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str',
    'boolean': 'https://docs.python.org/3/library/stdtypes.html#boolean-values',
    'tuple': 'https://docs.python.org/3/library/stdtypes.html#tuples',
    'list': 'https://docs.python.org/3/library/stdtypes.html#lists',
    'dict': 'https://docs.python.org/3/library/stdtypes.html#mapping-types-dict',
    'namedtuple': 'https://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields',
    'callable': 'https://docs.python.org/3/library/functions.html#callable',
    'dataclass': 'https://docs.python.org/3/library/dataclasses.html',
    # Python constants.
    'False': 'https://docs.python.org/3/library/constants.html#False',
    'True': 'https://docs.python.org/3/library/constants.html#True',
    'None': 'https://docs.python.org/3/library/constants.html#None',
    # NumPy types.
    'np.ndarray': 'https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html',
    'np.inf': 'https://numpy.org/doc/stable/reference/constants.html#numpy.inf',
    'np.nan': 'https://numpy.org/doc/stable/reference/constants.html#numpy.nan',
    # PyTorch types.
    'torch.Tensor': 'https://pytorch.org/docs/stable/tensors.html',
    'torch.Size': 'https://pytorch.org/docs/stable/size.html',
    'torch.dtype': 'https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype',
    'torch.device': 'https://pytorch.org/docs/stable/tensor_attributes.html#torch.device',
    # TorchGeometric types.
    'torch_geometric.data.Data': 'https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data',
}

IM2SIM_OBJECTS_PATTERN = re.compile(
    r"``(?P<name>im2sim\.[a-zA-Z0-9_.]+)``"
)

COMMON_TYPES_PATTERNS = {
    k: re.compile(rf"``{k}``")for k in COMMON_TYPES_LINKS}

COMMON_TYPES_REPLACEMENTS = {
    k: rf"`{k} <{v}>`_" for k, v in COMMON_TYPES_LINKS.items()}

CODE_LETTER_PATTERN = re.compile(r"``(?P<code>\w+)``(?P<letter>[a-zA-Z])")
CODE_LETTER_REPL = r"``\g<code>``\ \g<letter>"

LINK_PATTERN = re.compile(r"``(?P<link_text>[\w\.]+)``_")
LINK_REPL = r"`\g<link_text>`_"

import inspect


def process_docstring_text(text):
  """Process a docstring and convert it to Sphinx RST."""
  # Replace Note: and Warning: by RST equivalents.
  rst_lines = []
  admonition_lines = None

  for line in text.splitlines():
    if admonition_lines is None:
      # We are not in an admonition right now. Check if this line will start
      # one.
      if (line.strip().startswith('Warning:') or
          line.strip().startswith('Note:')):
        label_position = line.index(':')
        admonition_type = line[:label_position].strip().lower()
        admonition_content = line[label_position + 1:].strip()
        leading_whitespace = ' ' * (len(line) - len(line.lstrip()))
        extra_indentation = '  '

        admonition_lines = [
            f"{leading_whitespace}.. {admonition_type}::",
            leading_whitespace + extra_indentation + admonition_content,
        ]
      else:
        rst_lines.append(line)
    else:
      # Check if this is the end of the admonition.
      if line.strip() == '':
        rst_lines.extend(admonition_lines)
        admonition_lines = None
      else:
        admonition_lines.append(extra_indentation + line)

  # If we reached the end and are still in an admonition, add it.
  if admonition_lines is not None:
    rst_lines.extend(admonition_lines)

  # Replace markdown literal markers (`) by ReST literal markers (``).
  text = '\n'.join(rst_lines)
  text = text.replace('`', '``')
  text = text.replace(':math:``', ':math:`')

  # Correct inline code followed by word characters.
  text = CODE_LETTER_PATTERN.sub(CODE_LETTER_REPL, text)

  # Add links to common types.
  for k in COMMON_TYPES_LINKS:
    text = COMMON_TYPES_PATTERNS[k].sub(
        COMMON_TYPES_REPLACEMENTS[k],
        text,
    )


  # Add links to im2sim objects.
  for match in IM2SIM_OBJECTS_PATTERN.finditer(text):
    object_name = match.group('name')
    url = get_doc_url(object_name)

    pattern = rf"``{object_name}``"
    repl = rf"`{object_name} <{url}>`_"
    text = text.replace(pattern, repl)

  # Correct double quotes.
  text = LINK_PATTERN.sub(LINK_REPL, text)

  return text


def process_docstring(
    app, what, name, obj, options, lines
):  # pylint: disable=missing-param-doc,unused-argument
  """Process autodoc docstrings."""
  text = process_docstring_text('\n'.join(lines))
  lines[:] = text.splitlines()
  return
  # if what != 'class':
  #   return

  # presets = getattr(obj, '_presets', None)
  # if not presets:
  #   return

  # lines.append('')
  # lines.append('.. rubric:: Preset Library')
  # lines.append('')

  # for preset_name, fn in presets.items():
  #   doc = inspect.getdoc(fn) or 'No description provided.'

  #   lines.append(f'**{preset_name}**')
  #   lines.append('')

  #   # Process preset documentation using exactly the same rules
  #   # as the class documentation.
  #   preset_text = process_docstring_text(doc)

  #   lines.extend(preset_text.splitlines())
  #   lines.append('')


def get_doc_url(name):
  """Get doc URL for the given im2sim name."""
  url = 'https://mrphys.github.io/im2sim/api_docs/'
  url += name.replace('.', '/')
  return url


def setup(app):
  app.add_css_file('custom.css')
  app.connect('autodoc-process-docstring', process_docstring)
