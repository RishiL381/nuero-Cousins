"""Load the EPIStress pandas pickles without letting them run code.

A pickle can call any Python function while loading, so a file from the internet could run commands.
This loader only allows the pandas and numpy classes that a pickled Series/DataFrame needs and
refuses everything else (os, subprocess, eval, ...).
"""

from pandas.compat.pickle_compat import Unpickler

ALLOWED_PREFIXES = ("pandas.", "numpy")
ALLOWED_BUILTINS = {"slice", "complex", "set", "frozenset", "bytearray", "range"}


class SafeUnpickler(Unpickler):
    def find_class(self, module, name):
        if module.startswith(ALLOWED_PREFIXES) or (module == "builtins" and name in ALLOWED_BUILTINS):
            return super().find_class(module, name)
        raise ValueError(f"blocked {module}.{name} while loading a pickle")


def load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()
