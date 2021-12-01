import random
from typing import Dict, List, Optional

import numpy as np
from numpy.lib.npyio import NpzFile


class ArrayArchive:
    """A dictionary-like storage for numpy arrays. All keys must be strings.

    ArrayArchive's can be easily saved and loaded from disc using the `*.npz`-file
    functionality supported by `numpy`. Different to the native implementation, this
    supports a more convenient interface, especially if working with the archived arrays
    *offline* -- i.e., while there is no open connection to the archive in the local
    storage, and all numpy arrays are in memory.
    """

    _npzfile: Optional[NpzFile] = None

    def __init__(self, arrays: Optional[Dict[str, np.ndarray]] = None) -> None:
        self._arrays = arrays or {}
        self._npzfile = None  # offline mode

    def save(self, path: str) -> None:
        """Save this ArrayArchive to path"""
        if self._npzfile:
            for key in self._npzfile.files:
                self._arrays[key] = self._npzfile[key]  # type: ignore

        np.savez(file=path, **self._arrays)

    def load(self, path: str, into_memory: bool = True):
        """Load an ArrayArchive from path. Returns self.

        * Offline usage (`into_memory=True`): all data loaded into memory, default
        * Online usage (`into_memory=False`): data loaded only once needed, connection
        to the file needs to be closed either manually or used a context manager, e.g.:
        `with ArrayArchive().load(path) as archive: ...`
        """
        self._npzfile = np.load(path, allow_pickle=True)  # type: ignore
        if into_memory:
            self._load_into_memory()
        return self

    def _load_into_memory(self) -> None:
        assert self._npzfile is not None
        for key in self._npzfile.files:
            self._arrays[key] = self._npzfile[key]  # type: ignore
        self.close()

    def close(self) -> None:
        """Close connection to a disc file (online usage)"""
        if self._npzfile:
            self._npzfile.close()
            self._npzfile = None

    @property
    def keys(self) -> List[str]:
        """All keys to the arrays in this ArrayArchive"""
        keys = list(self._arrays.keys())
        if self._npzfile:
            keys += self._npzfile.files
        return keys

    def to_list(self, shuffle: bool = False) -> List[np.ndarray]:
        """Return archive content arrays as a list"""
        response = [self[key] for key in self.keys]
        if shuffle:
            random.shuffle(response)
        return response

    def __len__(self) -> int:
        return len(self.keys)

    def __repr__(self) -> str:
        connection = "online" if self._npzfile else "offline"
        return f"<{self.__class__.__name__} with {len(self)} keys - {connection}>"

    def __getitem__(self, key: str) -> np.ndarray:
        if self._npzfile and key in self._npzfile.files:
            return self._npzfile[key]  # type: ignore
        else:
            return self._arrays[key]

    def __setitem__(self, key: str, array: np.ndarray) -> None:
        if not isinstance(key, str):
            raise KeyError(f"key should be a string, not {type(key)}")
        if self._npzfile and key in self._npzfile.files:
            raise Exception("can't update value from disc in online mode")
        self._arrays[key] = array

    def __delitem__(self, key: str) -> None:
        if self._npzfile and key in self._npzfile.files:
            self._load_into_memory()  # can't delete from archive on disc
        else:
            del self._arrays[key]

    def pop(self, key: str) -> np.ndarray:
        """Get array with this key and remove it from memory. Enforces offline mode."""
        if self._npzfile:
            self._load_into_memory()
        array = self[key]
        del self[key]
        return array
