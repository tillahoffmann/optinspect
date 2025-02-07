import doctest


project = "optinspect"
html_theme = "sphinx_book_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
]
doctest_default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
