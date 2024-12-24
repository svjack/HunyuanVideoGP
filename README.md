<!-- ## **HunyuanVideo** -->

[中文阅读](./README_zh.md)


# HunyuanVideo: A Systematic Framework For Large Video Generation Model
<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2412.03603"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv:HunyuanVideo&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp; &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-PromptRewrite"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-PromptRewrite&message=HuggingFace&color=yellow"></a> &ensp; &ensp;

  [![Replicate](https://replicate.com/zsxkib/hunyuan-video/badge)](https://replicate.com/zsxkib/hunyuan-video)
</div>



*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly on a 12 GB to 24 GB GPU.*

This version has the following improvements over the original Hunyuan Video model:
- Reduce greatly the RAM requirements and VRAM requirements
- 5 profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM) 
- Autodownloading of the needed model files
- Improved gradio interface with progression bar and more options
- Switch easily between Hunyuan and Fast Hunyuan models and quantized / non quantized models
- Much simpler installation



This fork by DeepBeepMeep is an integration of the mmpg module on the gradio_server.py.

It is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with changing only a few lines of code in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp

You will find the original Hunyuan Video repository here: https://github.com/Tencent/HunyuanVideo
 


## **Abstract**
We present HunyuanVideo, a novel open-source video foundation model that exhibits performance in video generation that is comparable to, if not superior to, leading closed-source models. In order to train HunyuanVideo model, we adopt several key technologies for model learning, including data curation, image-video joint model training, and an efficient infrastructure designed to facilitate large-scale model training and inference. Additionally, through an effective strategy for scaling model architecture and dataset, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. 

We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion diversity, text-video alignment, and generation stability. According to professional human evaluation results, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and 3 top-performing Chinese video generative models. By releasing the code and weights of the foundation model and its applications, we aim to bridge the gap between closed-source and open-source video foundation models. This initiative will empower everyone in the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. 



### Installation Guide for Linux

We provide an `environment.yml` file for setting up a Conda environment.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

We recommend CUDA versions 12.4 or 11.8 for the manual installation.

```shell
# 1. Prepare conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate HunyuanVideo

# 3. Install pip dependencies
python -m pip install -r requirements.txt


# 4.1 optional Flash attention support (easy to install on Linux but much harder on Windows)
python -m pip install flash-attn==2.7.2.post1

# 4.2 optional Sage attention support (30% faster, easy to install on Linux but much harder on Windows)
python -m pip install sageattention==1.0.6 

```

### Profiles
You can choose between 5 profiles depending on your hardware:
- HighRAM_HighVRAM_Fastest (1): at least 48 GB of RAM and 24 GB of VRAM : the fastest well suited for a RTX 3090 / RTX 4090
- HighRAM_LowVRAM_Fast (2): at least 48 GB of RAM and 12 GB of VRAM : a bit slower, better suited for RTX 3070/3080/4070/4080 
            or for RTX 3090 / RTX 4090 with large pictures batches or long videos
- LowRAM_HighVRAM_Medium (3): at least 32 GB of RAM and 24 GB of VRAM : so so speed but adapted for RTX 3090 / RTX 4090 with limited RAM
- LowRAM_LowVRAM_Slow (4): at least 32 GB of RAM and 12 GB of VRAM :  if you have little VRAM or generate longer videos 
- VerylowRAM_LowVRAM_Slowest (5): at least 24 GB of RAM and 10 GB of VRAM : if you don't have much it won't be fast but maybe it will work

If you use a Windows system, you will need 16 extra GB of RAM due to restrictions on available reserved memory.\
A safe approach is to start from profile 5 or 4 and go down progressively to 1 as long as the app remains responsive or doesn't trigger any out of memory error.


### Run a Gradio Server on port 7860 (recommended)
```bash
python3 gradio_server.py
```

You will have the possibility to configure a RAM / VRAM profile by expanding the section *Video Engine Configuration* in the Web Interface.\
If by mistake you have chosen a configuration not supported by your system, you can force a profile while loading the app with safe profile values such as 4 or 5:  
```bash
python3 gradio_server.py --profile 4
```


### Run through the command line
```bash
cd HunyuanVideo

python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --save-path ./results
```

Please note currently that profile and the models used need to be mentioned inside the *sample_video.py* file.

### More Configurations

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--prompt`       |   None    |   The text prompt for video generation    |
|     `--video-size`     | 720 1280  |      The size of the generated video      |
|    `--video-length`    |    129    |     The length of the generated video     |
|    `--infer-steps`     |    50     |     The number of steps for sampling      |
| `--embedded-cfg-scale` |    6.0    |    Embeded  Classifier free guidance scale       |
|     `--flow-shift`     |    7.0    | Shift factor for flow matching schedulers |
|     `--flow-reverse`   |    False  | If reverse, learning/sampling from t=1 -> t=0 |
|        `--seed`        |     None  |   The random seed for generating video, if None, we init a random seed    |
|  `--use-cpu-offload`   |   False   |    Use CPU offload for the model load to save more memory, necessary for high-res video generation    |
|     `--save-path`      | ./results |     Path to save the generated video      |



## 🔗 BibTeX
If you find [HunyuanVideo](https://arxiv.org/abs/2412.03603) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv preprint arXiv:2412.03603},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03603}, 
}
```



## 🧩 Projects that use HunyuanVideo

If you develop/use HunyuanVideo in your projects, welcome to let us know.

- ComfyUI (with support for F8 Inference and Video2Video Generation): [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) by [Kijai](https://github.com/kijai)



## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
Additionally, we also thank the Tencent Hunyuan Multimodal team for their help with the text encoder. 

## Star History
<a href="https://star-history.com/#Tencent/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
 </picture>
</a>
