# BitCore

BitCore provides quantization-aware binary linear layers that can swap into
deployment mode using the accompanying `bitops` extension for efficient
inference. Install in editable mode during development and use `BitLinear`
from the package namespace:

```bash
pip install -e .
```

```python
from bitcore import BitLinear
```

