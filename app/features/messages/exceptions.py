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


class TurnFailedError(AppError):
    status = 500
    code = Code.TURN_FAILED
