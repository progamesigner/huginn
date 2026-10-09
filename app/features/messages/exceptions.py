from app.core.codespaces import Code
from app.core.exceptions import AppError


class ForbiddenAgentError(AppError):
    status = 403
    code = Code.FORBIDDEN_AGENT


class ConversationBusyError(AppError):
    status = 409
    code = Code.CONVERSATION_BUSY


class UnroutableRequestError(AppError):
    status = 400
    code = Code.UNROUTABLE_REQUEST


class AttachmentTooLargeError(AppError):
    status = 413
    code = Code.ATTACHMENT_TOO_LARGE


class UnsupportedAttachmentError(AppError):
    status = 415
    code = Code.UNSUPPORTED_ATTACHMENT


class InvalidAttachmentError(AppError):
    status = 400
    code = Code.INVALID_ATTACHMENT


class BlobNotFoundError(AppError):
    status = 404
    code = Code.BLOB_NOT_FOUND


class TurnFailedError(AppError):
    status = 500
    code = Code.TURN_FAILED
