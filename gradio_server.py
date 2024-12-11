import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import random

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import NEGATIVE_PROMPT

args = parse_args()
models_root_path = Path(args.model_base)
if not models_root_path.exists():
    raise ValueError(f"`models_root` not exists: {models_root_path}")

import json

with open("./ckpts/hunyuan-video-t2v-720p/vae/config.json", "r", encoding="utf-8") as reader:
    text = reader.read()
vae_config= json.loads(text)
# reduce time window
if vae_config["sample_tsize"] == 64:
    vae_config["sample_tsize"] = 32 
with open("./ckpts/hunyuan-video-t2v-720p/vae/config.json", "w", encoding="utf-8") as writer:
    writer.write(json.dumps(vae_config))

args.flow_reverse = True    

hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args, device="cpu")
from mmgp import offload 
pipe = hunyuan_video_sampler.pipeline
#offload.all(pipe, pinInRAM=True) # faster but you need at least 64 GB of RAM
offload.all(pipe, pinInRAM=False)

def generate_video(
    prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    progress=gr.Progress(track_tqdm=True)

):
    seed = None if seed == -1 else seed
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    negative_prompt = "" # not applicable in the inference

    outputs = hunyuan_video_sampler.predict(
        prompt=prompt,
        height=height,
        width=width, 
        video_length=(video_length // 4)* 4 + 1 ,
        seed=seed,
        negative_prompt=negative_prompt,
        infer_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=1,
        flow_shift=flow_shift,
        batch_size=1,
        embedded_guidance_scale=embedded_guidance_scale
    )
    
    samples = outputs['samples']
    sample = samples[0].unsqueeze(0)
    
    save_path = os.path.join(os.getcwd(), "gradio_outputs")
    os.makedirs(save_path, exist_ok=True)
    
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    video_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][0]}_{outputs['prompts'][0][:100].replace('/','')}.mp4"
    save_videos_grid(sample, video_path, fps=24)
    logger.info(f'Sample saved to: {video_path}')
    
    return video_path

def create_demo(model_path, save_path):
    
    with gr.Blocks() as demo:
        gr.Markdown("<div align=center><H1>HunyuanVideo<SUP>GP</SUP> by Tencent</H3></div>")
        gr.Markdown("*GPU Poor version by **DeepBeepMeep**. Now this great video generator can run smoothly on a 24 GB rig.*")
        gr.Markdown("Please be aware of these limits if you have a RTX 3090 / RTX 4090:")
        gr.Markdown("- max 97 frames for 848 x 480")
        gr.Markdown("- max 41 frames for 1280 x 720")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect.")
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("1280x720 (16:9, 720p)", "1280x720"),
                            ("720x1280 (9:16, 720p)", "720x1280"), 
                            ("1104x832 (4:3, 720p)", "1104x832"),
                            ("832x1104 (3:4, 720p)", "832x1104"),
                            ("960x960 (1:1, 720p)", "960x960"),
                            # 540p
                            ("960x544 (16:9, 540p)", "960x544"),
                            ("848x480 (16:9, 540p)", "848x480"),
                            ("544x960 (9:16, 540p)", "544x960"),
                            ("832x624 (4:3, 540p)", "832x624"), 
                            ("624x832 (3:4, 540p)", "624x832"),
                            ("720x720 (1:1, 540p)", "720x720"),
                        ],
                        value="848x480",
                        label="Resolution"
                    )

                video_length = gr.Slider(5, 133, value=97, step=4, label="Number of frames (30 = 1s)")

                    # video_length = gr.Dropdown(
                    #     label="Video Length",
                    #     choices=[
                    #         ("1.5s(41f)", 41),
                    #         ("2s(65f)", 65),
                    #         ("4s(97f)", 97),
                    #         ("5s(129f)", 129),
                    #     ],
                    #     value=97,
                    # )
                num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Number of Inference Steps")
                show_advanced = gr.Checkbox(label="Show Advanced Options", value=False)
                with gr.Row(visible=False) as advanced_row:
                    with gr.Column():
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        guidance_scale = gr.Slider(1.0, 20.0, value=1.0, step=0.5, label="Guidance Scale")
                        flow_shift = gr.Slider(0.0, 10.0, value=7.0, step=0.1, label="Flow Shift") 
                        embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale")
                show_advanced.change(fn=lambda x: gr.Row(visible=x), inputs=[show_advanced], outputs=[advanced_row])
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                output = gr.Video(label="Generated Video")
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                guidance_scale,
                flow_shift,
                embedded_guidance_scale
            ],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "7860"))
    demo = create_demo(args.model_base, args.save_path)
    demo.launch(server_name=server_name, server_port=server_port)