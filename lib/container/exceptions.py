class ContainerError(Exception):
    pass


class BindingError(ContainerError):
    pass


class ResolutionError(ContainerError):
    pass
