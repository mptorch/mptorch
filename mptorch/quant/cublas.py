class cublas_acceleration:
    r"""cuBLAS acceleration management.

    This class allows enabling and disabling of automatic cuBLAS acceleration
    for compatible types. When enabled, all calls for float quantized (batched) GEMMs
    (`float_mm` and `float_bmm`) used internally linear and convolutional layers will
    use the cuBLAS GEMM functions if the floating point computation formats matches one
    of the formats combinations supported by cuBLAS, i.e:
    - nearest rounding even mode
    - fused-multiply-add enabled
    - subnormals enabled
    - saturation disabled
    - same multiplication/accumulator types matching one of the
    [supported combinations](https://docs.nvidia.com/cuda/cublas/#cublasgemmex)

    This feature is disabled by default.

    Args:
        enabled: whether to enable automatic cuBLAS acceleration.
        fast_mode: allow internal downcast to lower-precision for tensor
            cores. Currently supported are `f16`, `bf16` and `tf32`

    Example:
        cuBLAS acceleration can be enabled/disabled globally using the static
        `enable` method; or locally as a context manager:

        .. code-block:: python

            mac_format = FloatingPoint(
                exp=5, man=10, subnormals=True, saturate=False # F16, supported by cublas
            )
            layer_formats = QAffineFormats(
                fwd_mac=(mac_format,),
                bwd_mac=(mac_format,),
                fwd_rnd="nearest",
                bwd_rnd="nearest",
                ...
            )
            layer = QLinear(in_features, out_features, formats=layer_formats)
            with cublas_acceleration(True):
                x = torch.tensor(...)
                y = layer.forward(x)
    """

    enabled = False
    fast_mode = None

    def __init__(self, enabled: bool, fast_mode: str | None = None):
        self._enter_status = enabled
        self._enter_fast_mode = fast_mode

    @classmethod
    def enable(cls, status: bool, fast_mode: str | None = None):
        """Globally enables or disables cuBLAS acceleration.

        Args:
            status: whether to enable or disable cuBLAS acceleration
            fast_mode: use down-conversion to `f16`, `bf16` or `tf32` for faster GEMM when possible
        """
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
