# LISA

[![arXiv](https://img.shields.io/badge/arXiv-2409.05901-b31b1b.svg)](https://arxiv.org/abs/2602.04906)

how to use:
```python
import os, sys, subprocess as sbp
import numpy as np
import matplotlib.pyplot as plt
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
try:
    from lisa import LISA, ALSA
except:
    sbp.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/sparsetrace/LISA.git"])
    from lisa import LISA, ALSA
```
