# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'TimeSeAD'
copyright = '2023, TimeSeAD authors'
author = 'Tobias Michels, Arjun Nair, Florian C.F. Schulz, Dennis Wagner'

release = '0.1'
version = '0.1.0'

# -- General configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'myst_parser',
    # 'autodoc2',
    'autoapi.extension',
]

suppress_warnings = [
    'ref.citation',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sacred': ('https://sacred.readthedocs.io/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

autosummary_generate = False
# autosummary_imported_members = True

# autodoc_default_options = {
#     'members': True,
# }
autodoc_typehints = 'both'

autoapi_dirs = ['../../timesead']
autoapi_options = [
    'members',
    'undoc-members',
    # 'private-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]
autoapi_keep_files = True
autoapi_python_class_content = 'both'
