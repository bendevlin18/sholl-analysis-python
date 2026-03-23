"""
supported_formats.py
--------------------
Central definition of supported image file extensions.
Add new formats here and they will be picked up everywhere automatically.
"""

# All extensions the package will recognise, lowercase, with leading dot
SUPPORTED_EXTENSIONS = (".tiff", ".tif", ".png", ".jpg", ".jpeg")


def is_supported(filename: str) -> bool:
    """Return True if *filename* has a recognised image extension."""
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)


def get_stem(filename: str) -> str:
    """
    Strip the file extension and return the stem.

    Works correctly for multi-dot extensions like ``.tiff``.

    Examples
    --------
    >>> get_stem("cell_01.tiff")
    'cell_01'
    >>> get_stem("cell_01.tif")
    'cell_01'
    >>> get_stem("cell_01.png")
    'cell_01'
    """
    for ext in SUPPORTED_EXTENSIONS:
        if filename.lower().endswith(ext):
            return filename[: -len(ext)]
    # Fallback — strip whatever extension is present
    return filename.rsplit(".", 1)[0]


def extensions_str() -> str:
    """Human-readable list of supported extensions for help text / errors."""
    return ", ".join(SUPPORTED_EXTENSIONS)
