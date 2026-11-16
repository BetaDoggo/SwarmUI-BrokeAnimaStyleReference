# Broke Anima Style Reference SwarmUI

A shitty vibed up training-free style reference method for anima. Only tested on turbo because I'm lazy.

<img width="1431" height="1067" alt="Screenshot 2026-11-16 133538" src="https://github.com/user-attachments/assets/c4b8e851-5955-4489-9ec1-e98d033c2391" />


## How it works

At each sampling step, the picture is noised to that step's noise level and run through the unmodified model with a
neutral caption. After every transformer block, each of the generated image's 2048 channels moves part of the way
(**Style Match**) towards the picture's mean and standard deviation for that channel at the same block. This is AdaIN
inside the DiT. Only per-channel statistics cross over, with no positions, so no object from the picture can. It
costs one extra model pass per step.

* **Style Match** (0 to 1, default 0.6): how far the statistics move. Near 1 the image falls apart.
* **Style Restyle** (default on): the first quarter of the steps run without the picture, so the image keeps the
  composition the prompt would have had. Off gives a stronger style, with a composition that drifts towards the picture's.
* **Style Megapixels** (0.5 to 4, default 1): the picture is shrunk to this area first, keeping its aspect ratio
  (never enlarged). About the generated image's own size works best (1 for 1024x1024). Smaller loses coarse texture
  such as pixel-art blocks. Larger drifts off the style, and at 4 with a 1024x1024 image the result falls apart.
  Bigger is also slower: at 1024x1024 the extra pass adds about half a step at 0.5 and a whole step at 1.

## Install

Just copy this folder into `SwarmUI/src/Extensions/` and restart SwarmUI. The comfy portion is included, though you will need to put it into your custom nodes folder manually if you're using a remote comfy.

It's also compatible with my rust backend: https://github.com/BetaDoggo/comfyui-backend-but-rust

## Here's the work the llm probably stole from:

* **Style as feature statistics.** Gatys, Ecker and Bethge showed that an image's style lives in the statistics of a
  network's features rather than in where things are. Li et al. showed that matching per-channel means and variances
  is enough.
  * L. A. Gatys, A. S. Ecker, M. Bethge. *Image Style Transfer Using Convolutional Neural Networks.* CVPR 2016.
    [CVF open access](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)
  * Y. Li, N. Wang, J. Liu, X. Hou. *Demystifying Neural Style Transfer.* IJCAI 2017. [arXiv:1701.01036](https://arxiv.org/abs/1701.01036)
* **AdaIN.** Huang and Belongie's adaptive instance normalization re-normalizes content features to a style image's
  per-channel mean and standard deviation. This method is AdaIN applied to a DiT's residual stream after every block,
  blended in by Style Match.
  * X. Huang, S. Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* ICCV 2017.
    [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
* **Training-free style references in diffusion models.** StyleAligned shares self-attention with a reference
  and applies AdaIN to its queries and keys. StyleID injects a style image's attention features and uses AdaIN on the
  initial latent. Shared attention was tried here first and dropped, because on Anima it copies the reference's subject.
  The AdaIN half is what remained.
  * A. Hertz, A. Voynov, S. Fruchter, D. Cohen-Or. *Style Aligned Image Generation via Shared Attention.* CVPR 2024.
    [arXiv:2312.02133](https://arxiv.org/abs/2312.02133)
  * J. Chung, S. Hyun, J.-P. Heo. *Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale
    Diffusion Models for Style Transfer.* CVPR 2024. [arXiv:2312.09008](https://arxiv.org/abs/2312.09008)
* **A noised reference beside the generation, and AdaIN against it.** Running the reference through the same
  model at each step follows lllyasviel's "reference-only" control for Stable Diffusion. Its `reference_adain`
  preprocessor added AdaIN against that reference, which makes it the closest prior art. This node is essentially
  `reference_adain` moved to Anima's DiT blocks. SwarmUI has a similar "Use Reference Only" option which only applied to Unet models.
  * lllyasviel. *[Major Update] Reference-only Control.* sd-webui-controlnet, May 2023.
    [Discussion #1236](https://github.com/Mikubill/sd-webui-controlnet/discussions/1236)
  * lllyasviel. *[New Preprocessor] The "reference_adain" and "reference_adain+attn" are added.* sd-webui-controlnet
    v1.1.171, May 2023. [Discussion #1280](https://github.com/Mikubill/sd-webui-controlnet/discussions/1280)
