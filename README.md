# PhotoGuard: Defending Against Diffusion-based Image Manipulation


This repository contains the code for our recent work on safe-guarding images against manipulation by ML-powerd photo-editing models such as [stable diffusion](https://stability.ai/blog/stable-diffusion-public-release).

**PhotoGuard: Defending Against Diffusion-based Image Manipulation** <br>
*Hadi Salman\*, Alaa Khaddaj\*, Guillaume Leclerc\*, Andrew Ilyas\*, Aleksander Madry* <br>
**Paper:** Coming soon! <br>
**Blog post:** https://gradientscience.org/photoguard/  <br>

```bibtex
    @article{salman2022photoguard,
        title={PhotoGuard: Defending Against Diffusion-based Image Manipulation},
        author={Hadi Salman and Alaa Khaddaj and Guillaume Leclerc and Andrew Ilyas and Aleksander Madry},
        year={2022},
        booktitle={Github Repository}
    }
```

<p>
<img src="assets/hero_fig.png" width="1000" >
</p>

## Getting started
Our code relies on the [stable diffusion code on Hugging Face](https://github.com/huggingface/diffusers). 

1.  Clone our repo: `git clone https://github.com/madrylab/photoguard.git`

2.  Install dependencies:
    ```
      conda create -n photoguard python=3.10
      conda activate photoguard
      pip install --upgrade diffusers transformers scipy
      huggingface-cli login
    ```

3. You should now be all set! Check out our notebooks!


## Generating high-quality fake images
The first step is we will walk you through how you can generate high quality fake images. Check out this notebook! The result will be such images:

See [this notebook](notebooks/generating_fake_images.ipynb)!

<p>
<img src="assets/hadi_trevor.png" width="1000" >
</p>


## Simple photo-guarding:
Now, we describe the simplest form of photo safeguarding that we implement. In particular, we implement a simple PGD attack on the image embedding part of the stable diffusion model. We have two demos demonstrating the efficacy of such photo safeguarding method. The goal of both is to cause the stable diffusion model to generate something that is either unrealistic, or unrelated to the original image.

### Photo-guarding against Image-to-Image pipelines
The first is the case where someone uses an image + prompt to modify the input image based on the prompt description.

See [this notebook](notebooks/demo_simple_attack_img2img.ipynb)!
<p>
<img src="assets/simple_attack_img2img.png" width="1000" >
</p>


### Photo-guarding against Inpainting pipelines
The second is the more interesting scenario where someone wants to edit parts of an existing image via inpainting. The generated images after immunization are clearly fake!

See [this notebook](notebooks/demo_simple_attack_inpainting.ipynb)!
<p>
<img src="assets/simple_attack_inpaint.png" width="1000" >
</p>


## Complex photo-guarding
For more effective photo-guarding especially against image inpainting, we need to attack the stable diffusion model end-to-end. Now, the generated images after immunization are even more clearly fake than above!

See [this notebook](notebooks/demo_complex_attack_inpainting.ipynb)!

<p>
<img src="assets/complex_attack_inpaint.png" width="1000" >
</p>



That's it! Please let us know if you have any questions.

# Maintainers

* [Hadi Salman](https://twitter.com/hadisalmanX)
* [Alaa Khaddaj](https://twitter.com/Alaa_Khaddaj) 
* [Guillaume Leclerc](https://twitter.gcom/gpoleclerc) 
* [Andrew Ilyas](https://twitter.com/andrew_ilyas)
* [Aleksander Madry](https://twitter.com/aleks_madry) 
