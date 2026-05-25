#include "mm_ops.h"

MmDims infer_mm_dims(const Tensor &a, const Tensor &b)
{
  MmDims dims{};
  if (a.dim() > 2)
  {
    dims.batch = a.size(0);
    dims.M = a.size(1);
    dims.K = a.size(2);
    dims.N = b.size(2);
  }
  else
  {
    dims.batch = 1;
    dims.M = a.size(0);
    dims.K = a.size(1);
    dims.N = b.size(1);
  }
  return dims;
}
