"""OpenSteward application package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("opensteward")
except PackageNotFoundError:
    # package not installed, e.g. running from source without install
    __version__ = "unknown"