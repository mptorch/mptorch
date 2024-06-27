class cublas_acceleration:
    enabled = False

    def __init__(self, enabled: bool) -> None:
        self._enter_status = enabled
    
    @classmethod
    def enable(cls, status: bool):
        cls.enabled = status
    
    def __enter__(self):
        self._prev_status = self.__class__.enabled
        self.__class__.enabled = self._enter_status
    
    def __exit__(self, type, value, trace):
        self.__class__.enabled = self._prev_status
