"""Public, non-sensitive errors shared by the HTTP and graph boundaries."""


class ServiceError(RuntimeError):
    def __init__(self, code: str, status: int = 409):
        super().__init__(code)
        self.code = code
        self.status = status


class NodeExecutionError(RuntimeError):
    def __init__(self, node: str, cause_type: str):
        self.node = node
        self.cause_type = cause_type
        super().__init__(f"node_failed:{node}:{cause_type}")
