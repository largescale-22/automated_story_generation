# minDALL-E on Conceptual Captions

`minDALL-E`, named after [minGPT](https://github.com/karpathy/minGPT), is a 1.3B text-to-image generation model trained on 14 million
image-text pairs for non-commercial purposes.

## required files
- ckpt1, ckpt2
- fine-tuning data
- inference text

## Environment Setup
- Basic setup
```
PyTorch == 1.8.0
CUDA >= 10.1
```
- Other packages
```
pip install -r requirements.txt
```

## Model Checkpoint
- Model structure (two-stage autoregressive model)
  - Stage1: Unlike the original DALL-E [1], we replace Discrete VAE with VQGAN [2] to generate high-quality samples effectively.
            We slightly fine-tune [vqgan_imagenet_f16_16384](https://github.com/CompVis/taming-transformers), provided by the official VQGAN repository, on FFHQ [3] as well as ImageNet.
  - Stage2: We train our 1.3B transformer from scratch on 14 million image-text pairs from CC3M [4] and CC12M [5]. For the more detailed model spec, please see [configs/dalle-1.3B.yaml](configs/dalle-1.3B.yaml).
- You can download the pretrained models including the tokenizer from [this link](https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz). This will require about 5GB space.

## Sampling
- Given a text prompt, the code snippet below generates candidate images and re-ranks them using OpenAI's CLIP [6].
- This has been tested under a single V100 of 32GB memory. In the case of using GPUs with limited memory, please lower down num_candidates to avoid OOM.
```python
from matplotlib import pyplot as plt
import clip
from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score

device = 'cuda:0'
set_seed(0)

prompt = "A painting of a monkey with sunglasses in the frame"
model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
model.to(device=device)

# Sampling
images = model.sampling(prompt=prompt,
                        top_k=256, # It is recommended that top_k is set lower than 256.
                        top_p=None,
                        softmax_temperature=1.0,
                        num_candidates=96,
                        device=device).cpu().numpy()
images = np.transpose(images, (0, 2, 3, 1))

# CLIP Re-ranking
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.to(device=device)
rank = clip_score(prompt=prompt,
                  images=images,
                  model_clip=model_clip,
                  preprocess_clip=preprocess_clip,
                  device=device)

- If you want to use a complete python code for sampling, please see [examples/sampling_ex.py](examples/sampling_ex.py).
- If you want to play with an interactive demo, please see [examples/sampling_interactive_demo.ipynb](examples/sampling_interactive_demo.ipynb).
  Before using this, you may need to install [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/user_install.html).

## Licenses
* The `source codes` are licensed under [Apache 2.0](LICENSE.apache-2.0) License.
* The `stage2 pretrained weights` are licensed under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License.

## Limitations
Although `minDALL-E` is trained on a small set (14M image-text pairs), this might be vulnerable to malicious attacks from the prompt engineering to generate socially unacceptable images. If you obersve these images, please report the "prompt" and "generated images" to us.
