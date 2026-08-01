from app.core.exceptions import AppError


class ForbiddenAgentError(AppError):
    status = 403
    code = 2


class ConversationBusyError(AppError):
    status = 409
    code = 3


class UnroutableRequestError(AppError):
    status = 400
    code = 4


class TurnFailedError(AppError):
    status = 500
    code = 5
