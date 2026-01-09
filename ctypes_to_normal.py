import ctypes
from typing import Any

import numpy as np


def ctypes_to_normal(item: Any) -> Any:
    """
    Recursively converts ctypes objects to normal python or numpy objects.

    Pointers are dereferenced. Bytes are decoded to strings, if possible.
    Arrays are converted to numpy.ndarrays. Structures are converted to
    dictionaries. Returns input unaltered if it cannot be converted. Uses
    recursion heavily.

    Parameters
    ----------
    item : Any
        The item to be converted from a ctype to a normal python object.

    Returns
    -------
    Any type except a ctype
        The item as converted, if possible.
    """

    # Dereference pointers
    if isinstance(item, ctypes._Pointer):
        try:
            return ctypes_to_normal(item.contents)
        except ValueError:
            return None

    # Decode bytes
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError:
            return item

    # Convert Arrays to numpy ndarrays
    if isinstance(item, ctypes.Array):
        if issubclass(item._type_, ctypes.Structure):
            return np.array([ctypes_to_normal(x) for x in item])
        return np.ctypeslib.as_array(item).copy()

    # Iterate over struct fields to build dict
    if hasattr(item, "_fields_"):
        output = {}
        for field in item._fields_:
            key = ctypes_to_normal(field[0])
            value = getattr(item, key)
            output[key] = ctypes_to_normal(value)

        return output

    # Convert standard types
    if hasattr(item, "value"):
        return item.value

    # Return item unchanged
    return item
