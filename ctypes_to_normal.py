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

    # Decode bytes and and string pointers
    # c_char_p is not actually an instance of the _Pointer class
    if isinstance(item, (bytes, ctypes.c_char_p)):
        val = item.value if isinstance(item, ctypes.c_char_p) else item
        if val is None:
            return None
        try:
            # strip the binary null terminator
            return val.decode("utf-8", errors="ignore").strip("\x00")
        except (UnicodeDecodeError, AttributeError):
            return val

    # Convert Arrays to numpy ndarrays
    if isinstance(item, ctypes.Array):
        # SPECIAL CASE: convert array of c_char to string
        if item._type_ == ctypes.c_char:
            try:
                return item.value.decode("utf-8", errors="ignore")
            except Exception:
                # default to numpy array
                pass

        # Recurse on elements
        if issubclass(item._type_, (ctypes.Structure, ctypes.Array)):
            return np.array([ctypes_to_normal(x) for x in item])

        return np.ctypeslib.as_array(item).copy()

    # Union fields can reference overlapping memory segments
    # better to just return the object as is
    if isinstance(item, ctypes.Union):
        return item

    # Iterate over struct fields to build dict
    if isinstance(item, ctypes.Structure):
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
