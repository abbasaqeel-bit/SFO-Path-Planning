import numpy as np
from sfo_benchmark.core.path import Path

def test_path_length():
    p=Path(np.array([[0.,0.],[3.,4.]])); assert p.length()==5.0 and p.segment_count==1
