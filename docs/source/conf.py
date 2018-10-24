# -*- coding: utf-8 -*-
#
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
# rootdir = os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.abspath(__file__))))
# sys.path.append(os.path.join(rootdir, 'src'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]

# nbsphinx_allow_errors = True
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'ska-fastimgproto'
copyright = u'2016, Tim Staley'
author = u'Tim Staley'


# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

from fastimgproto import __version__ # isort:skip
version = __version__.split('+')[0]
if 'dirty' in version:
    version = version.rsplit('.', maxsplit=1)[0]
# The full version, including alpha/beta/rc tags.
# release = __versiondict__['full-revisionid'][:8]

language = None

# add_function_parentheses = True
# add_module_names = True
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

autoclass_content = 'class'
# napoleon_use_ivar = True

# Warn when references cannot be resolved:
nitpicky = True
nitpick_ignore = [
    ("py:obj", "list"), # https://github.com/sphinx-doc/sphinx/issues/2688 (?)
    ("py:obj", "numpy.ndarray"),
    ("py:obj", "tqdm.tqdm"),
]


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

html_title = u'ska-fastimgproto v' + version

html_last_updated_fmt = ''


html_show_sphinx = False


# Output file base name for HTML help builder.
htmlhelp_basename = 'ska-fastimgprotodoc'

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/2.7', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None)
}

# For quick building of regular docs during development:
# nbsphinx_execute = 'never'

nbsphinx_timeout = 60
