# Kaggle setup

Run this in the first Kaggle notebook cell before starting the Gradio app:

```bash
!pip uninstall -y torchvision
!pip install -q faster-whisper
```

Then restart the Kaggle session/kernel if Kaggle reports that packages were already imported. Verify the runtime in a fresh cell:

```python
import torch
import torchaudio
import transformers
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.device_count())
```

## Why this is needed

The narration app does not use image models or torchvision. However, Transformers detects the installed torchvision package while importing `LlamaModel`. Kaggle can contain a torch/torchvision binary mismatch, causing this error before Chatterbox starts:

```text
RuntimeError: operator torchvision::nms does not exist
```

If torchvision is needed for another notebook workload, reinstall a matched PyTorch stack instead. For PyTorch 2.6, the official matching versions are torch 2.6.0, torchvision 0.21.0, and torchaudio 2.6.0; select the CUDA wheel index that matches the runtime. See the [official PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/) page.
