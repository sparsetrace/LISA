# LISA

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
