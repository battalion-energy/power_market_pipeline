from enum import Enum


class ProductContentType(str, Enum):
    BINARY = "BINARY"
    DATA = "DATA"

    def __str__(self) -> str:
        return str(self.value)
