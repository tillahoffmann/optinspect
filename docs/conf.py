import doctest


project = "optinspect"
html_theme = "sphinx_book_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
]
doctest_default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
add_module_names = False
