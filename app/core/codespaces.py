from enum import IntEnum, unique


# NOTE: every code the API answers with lives in this one table; the number is the
# code and the name is the error, so neither can drift from the other or repeat
@unique
class Code(IntEnum):
    UNKNOWN_ERROR = 0
    UNAUTHORIZED = 1
    FORBIDDEN_AGENT = 2
    CONVERSATION_BUSY = 3
    UNROUTABLE_REQUEST = 4
    TURN_FAILED = 5
    INVALID_REQUEST = 6
    BLOB_NOT_FOUND = 7
