class cublas_acceleration:
    enabled = False
    fast_mode = None

    def __init__(self, enabled: bool, fast_mode: str | None = None):
        self._enter_status = enabled
        self._enter_fast_mode = fast_mode
    
    @classmethod
    def enable(cls, status: bool, fast_mode: str | None = None):
        cls.enabled = status
        cls.fast_mode = fast_mode
    
    def __enter__(self):
        self._prev_status = self.__class__.enabled
        self._prev_fast_mode = self.__class__.fast_mode
        self.__class__.enabled = self._enter_status
        self.__class__.fast_mode = self._enter_fast_mode
    
    def __exit__(self, type, value, trace):
        self.__class__.enabled = self._prev_status
        self.__class__.fast_mode = self._prev_fast_mode
