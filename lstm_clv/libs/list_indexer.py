from typing import Any, List


class ListIndexer:
    """A wrapper to efficiently get the position index of a list entry
    ```
    values = [1, 2, 3, ...]
    values.index(1)  # O(n)

    indexer = ListIndexer(values)
    indexer[1]  # O(1)
    ```
    """

    def __init__(self, values: List[Any]) -> None:
        self.values = values
        self.indices = {value: idx for idx, value in enumerate(values)}

    def __getitem__(self, value: Any) -> int:
        return self.indices[value]
