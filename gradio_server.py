import os
import time
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import random
import json
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import NEGATIVE_PROMPT
from mmgp import offload, safetensors2, profile_type 


args = parse_args()

lora_preselected =args.lora_weight
lora_dir =args.lora_dir

lora_preseleted_multiplier = [float(i) for i in args.lora_multiplier ]
force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
quantizeTransformer = args.quantize_transformer

transformer_choices=["ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_bf16.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_quanto_int8.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"]
text_encoder_choices = ["ckpts/text_encoder/llava-llama-3-8b-v1_1_fp16.safetensors", "ckpts/text_encoder/llava-llama-3-8b-v1_1_quanto_int8.safetensors"]
server_config_filename = "gradio_config.json"

if not Path(server_config_filename).is_file():
    server_config = {"attention_mode" : "sdpa",  
                     "transformer_filename": transformer_choices[1], 
                     "text_encoder_filename" : text_encoder_choices[1],
                     "compile" : "",
                     "profile" : profile_type.LowRAM_LowVRAM }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)

transformer_filename = server_config["transformer_filename"]
text_encoder_filename = server_config["text_encoder_filename"]
attention_mode = server_config["attention_mode"]
profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
compile = server_config.get("compile", "")

#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_bf16.safetensors"
#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_quanto_int8.safetensors"
#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"

#text_encoder_filename = "ckpts/text_encoder/llava-llama-3-8b-v1_1_fp16.safetensors"
#text_encoder_filename = "ckpts/text_encoder/llava-llama-3-8b-v1_1_quanto_int8.safetensors"

#attention_mode="sage"
#attention_mode="flash"
#attention_mode = "sdpa"

def download_models(transformer_filename, text_encoder_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        if not "quanto" in filename:
            return [filename]        
        pos = filename.rfind(".")
        return [filename, filename[:pos] +"_map.json"]
    
    from huggingface_hub import hf_hub_download, snapshot_download    
    repoId = "DeepBeepMeep/HunyuanVideo" 
    sourceFolderList = ["text_encoder_2", "text_encoder", "hunyuan-video-t2v-720p/vae", "hunyuan-video-t2v-720p/transformers" ]
    fileList = [ [], ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"] + computeList(text_encoder_filename) , [],  computeList(transformer_filename) ]
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
             for onefile in files:      
                if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                    hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)


download_models(transformer_filename, text_encoder_filename) 


offload.default_verboseLevel = verbose_level
with open("./ckpts/hunyuan-video-t2v-720p/vae/config.json", "r", encoding="utf-8") as reader:
    text = reader.read()
vae_config= json.loads(text)
# reduce time window used by the VAE for temporal splitting (former time windows is too large for 24 GB) 
if vae_config["sample_tsize"] == 64:
    vae_config["sample_tsize"] = 32 
with open("./ckpts/hunyuan-video-t2v-720p/vae/config.json", "w", encoding="utf-8") as writer:
    writer.write(json.dumps(vae_config))

args.flow_reverse = True
if profile == 5:
    pinToMemory = False
    partialPinning = False
else:    
    pinToMemory =  True
    import psutil
    physical_memory= psutil.virtual_memory().total    
    partialPinning = physical_memory <= 2**30 * 32 


hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(transformer_filename, text_encoder_filename, attention_mode = attention_mode, pinToMemory = pinToMemory, partialPinning = partialPinning, args=args,  device="cpu") 
pipe = hunyuan_video_sampler.pipeline

#compile = "transformer"

# lora_weight =["ckpts/arny_lora.safetensors"] # 'ohwx person' ,; 'wick'
# lora_multi = [1.0]
loras =[]
loras_names = []
default_loras_choices = []
default_loras_multis_str = ""

if len(lora_preselected) > 0:
    loras += lora_preselected
    loras_multis = (lora_preseleted_multiplier + ([1.0] * len(loras)) ) [:len(loras)]
    default_loras_choices = [ str(i) for i in range(len(loras))]
    default_loras_multis_str = "_".join([str(el) for el in loras_multis])

if lora_dir != None:
    if not os.path.isdir(lora_dir):
        raise Exception("--lora-dir should be a path to a directory that contains Loras")
    
    import glob
    dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
    dir_loras.sort()
    loras += [element for element in dir_loras if element not in loras ]

if len(loras) > 0:
    loras_names = [ Path(lora).stem for lora in loras  ]
    offload.load_loras_into_model(pipe.transformer, loras,  activate_all_loras=False) #lora_multiplier,

offload.profile(pipe, profile_no= profile, compile = compile, quantizeTransformer = quantizeTransformer) 
def apply_changes(
                    transformer_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
):
    server_config = {"attention_mode" : attention_choice,  
                     "transformer_filename": transformer_choices[transformer_choice], 
                     "text_encoder_filename" : text_encoder_choices[text_encoder_choice],
                     "compile" : compile_choice,
                     "profile" : profile_choice }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))

    return "<h1>New Config file created. Please restart the Gradio Server</h1>"


from moviepy.editor import ImageSequenceClip
import numpy as np

def save_video(final_frames, output_path, fps=24):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path, verbose= False, logger = None)

def build_callback(state, pipe, progress, status, num_inference_steps):
    def callback(step_idx, t, latents):
        step_idx += 1         
        if state.get("abort", False):
            pipe._interrupt = True
            status_msg = status + " - Aborting"    
        elif step_idx  == num_inference_steps:
            status_msg = status + " - VAE Decoding"    
        else:
            status_msg = status + " - Denoising"   

        progress( (step_idx , num_inference_steps) , status_msg  ,  num_inference_steps)
            
    return callback

def abort_generation(state):
    if "in_progress" in state:
        state["abort"] = True
        return gr.Button(interactive=  False)
    else:
        return gr.Button(interactive=  True)

def refresh_gallery(state):
    file_list = state["file_list"]      
    return file_list
        
def finalize_gallery(state):
    choice = 0
    if "in_progress" in state:
        del state["in_progress"]
        choice = state.get("selected",0)
    time.sleep(0.2)
    return gr.Gallery(selected_index=choice), gr.Button(interactive=  True)

def select_video(state , event_data: gr.EventData):
    data=  event_data._data
    if data!=None:
        state["selected"] = data.get("index",0)
    return 


def generate_video(
    prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    tea_cache,
    loras_choices,
    loras_mult_choices,
    state,
    progress=gr.Progress() #track_tqdm= True

):
    

    if len(loras) > 0:
        def is_float(element: any) -> bool:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
        list_mult_choices_str = loras_mult_choices.split(" ")
        list_mult_choices_nums = []
        for i, mult in enumerate(list_mult_choices_str):
            mult = mult.strip()
            if not is_float(mult):                
                raise gr.Error("Lora Multiplier no {i+1} ({mult}) is invalid")
            list_mult_choices_nums.append(float(mult))
        if len(list_mult_choices_nums ) <len(loras_choices):
            list_mult_choices_nums  += [1.0] * ( len(loras_choices) - len(list_mult_choices_nums ) )

        offload.activate_loras(pipe.transformer, loras_choices, list_mult_choices_nums)

    seed = None if seed == -1 else seed
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    negative_prompt = "" # not applicable in the inference

    if "abort" in state:
        del state["abort"]
    state["in_progress"] = True
    state["selected"] = 0
 


   # TeaCache
    trans = hunyuan_video_sampler.pipeline.transformer.__class__
    trans.enable_teacache = tea_cache > 0
    if trans.enable_teacache:
        trans.num_steps = num_inference_steps
        trans.cnt = 0
        trans.rel_l1_thresh = 0.15 # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
        trans.accumulated_rel_l1_distance = 0
        trans.previous_modulated_input = None
        trans.previous_residual = None
 
    import random
    if seed == None or seed <0:
        seed = random.randint(0, 999999999)

    file_list = []
    state["file_list"] = file_list    
    from einops import rearrange
    save_path = os.path.join(os.getcwd(), "gradio_outputs")
    os.makedirs(save_path, exist_ok=True)
    prompts = prompt.replace("\r", "").split("\n")
    video_no = 0
    total_video =  repeat_generation * len(prompts)

    start_time = time.time()
    for prompt in prompts:
        for _ in range(repeat_generation):
            video_no += 1
            status = f"Video {video_no}/{total_video}"
            progress(0, desc=status + " - Encoding Prompt" )   
            
            callback = build_callback(state, hunyuan_video_sampler.pipeline, progress, status, num_inference_steps)
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
                embedded_guidance_scale=embedded_guidance_scale,
                callback = callback,
                callback_steps = 1,

            )

            samples = outputs['samples']
            
            if samples == None:
                end_time = time.time()
                yield f"Abortion Succesful. Total Generation Time: {end_time-start_time:.1f}s"
            else:
                idx = 0
                # just in case one day we will have enough VRAM for batch geeneration ...
                for i,sample in enumerate(samples):
                    # sample = samples[0]
                    video = rearrange(sample.cpu().numpy(), "c t h w -> t h w c")

                    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                    file_name = f"{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','').strip()}.mp4".replace(':',' ').replace('\\',' ')
                    idx = 0 
                    basis_video_path = os.path.join(os.getcwd(), "gradio_outputs", file_name)        
                    video_path = basis_video_path
                    while True:
                        if  not Path(video_path).is_file():
                            idx = 0
                            break
                        idx += 1
                        video_path = basis_video_path[:-4] + f"_{idx}" + ".mp4"


                    save_video(video, video_path )
                    print(f"New video saved to Path: "+video_path)
                    file_list.append(video_path)
                    if video_no < total_video:
                        yield  status
                    else:
                        end_time = time.time()
                        yield f"Total Generation Time: {end_time-start_time:.1f}s"
            seed += 1



def create_demo(model_path, save_path):
    
    with gr.Blocks() as demo:
        gr.Markdown("<div align=center><H1>HunyuanVideo<SUP>GP</SUP></H3></div>")
        gr.Markdown("*Original model by Tencent, GPU Poor version by DeepBeepMeep. Now this great video generator can run smoothly on a 24 GB rig.*")
        gr.Markdown("Please be aware of these limits with profiles 2 and 4 if you have 24 GB of VRAM (RTX 3090 / RTX 4090):")
        gr.Markdown("- max 192 frames for 848 x 480 ")
        gr.Markdown("- max 86 frames for 1280 x 720")
        gr.Markdown("In the worst case, one step should not take more than 2 minutes. If it is the case you may be running out of RAM / VRAM. Try to generate fewer images / lower res / a less demanding profile.")
        gr.Markdown("If you have a Linux / WSL system you may turn on compilation (see below) and will be able to generate an extra 30°% frames")

        with gr.Accordion("Video Engine Configuration", open = False):
            gr.Markdown("For the changes to be effective you will need to restart the gradio_server")

            with gr.Column():
                index = transformer_choices.index(transformer_filename)
                index = 0 if index ==0 else index

                transformer_choice = gr.Dropdown(
                    choices=[
                        ("Hunyuan Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 0),
                        ("Hunyuan Video quantized to 8 bits (recommended) - the default engine but quantized", 1),
                        ("Fast Hunyuan Video quantized to 8 bits - requires less than 10 steps but worse quality", 2), 
                    ],
                    value= index,
                    label="Transformer"
                 )
                index = text_encoder_choices.index(text_encoder_filename)
                index = 0 if index ==0 else index

                gr.Markdown("Note that even if you choose a 16 bits Llava model below, depending on the profile it may be automatically quantized to 8 bits on the fly")
                text_encoder_choice = gr.Dropdown(
                    choices=[
                        ("Llava Llama 1.1 16 bits - unquantized text encoder, better quality uses more RAM", 0),
                        ("Llava Llama 1.1 quantized to 8 bits - quantized text encoder, worse quality but uses less RAM", 1),
                    ],
                    value= index,
                    label="Text Encoder"
                 )
                attention_choice = gr.Dropdown(
                    choices=[
                        ("Scale Dot Product Attention: default", "sdpa"),
                        ("Flash: good quality - requires additional install (usually complex to set up on Windows without WSL)", "flash"),
                        ("Sage: 30% faster but worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                    ],
                    value= attention_mode,
                    label="Attention Type"
                 )
                gr.Markdown("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation")
                compile_choice = gr.Dropdown(
                    choices=[
                        ("ON: works only on Linux / WSL", "transformer"),
                        ("OFF: no other choice if you have Windows without using WSL", "" ),
                    ],
                    value= compile,
                    label="Compile Transformer (up to 50% faster and 30% more frames but requires Linux / WSL and Flash or Sage attention)"
                 )                
                profile_choice = gr.Dropdown(
                    choices=[
                ("HighRAM_HighVRAM, profile 1: at least 48 GB of RAM and 24 GB of VRAM, the fastest for shorter videos a RTX 3090 / RTX 4090", 1),
                ("HighRAM_LowVRAM, profile 2 (Recommended): at least 48 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                ("LowRAM_HighVRAM, profile 3: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                ("LowRAM_LowVRAM, profile 4 (Default): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                ("VerylowRAM_LowVRAM, profile 5: (Fail safe): at least 16 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)
                    ],
                    value= profile,
                    label="Profile"
                 )

                msg = gr.Markdown()            
                apply_btn  = gr.Button("Apply Changes")

                apply_btn.click(
                        fn=apply_changes,
                        inputs=[
                            transformer_choice,
                            text_encoder_choice,
                            attention_choice,
                            compile_choice,                            
                            profile_choice,
                        ],
                        outputs= msg
                    )

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

                video_length = gr.Slider(5, 193, value=97, step=4, label="Number of frames (24 = 1s)")

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
    

                loras_choices = gr.Dropdown(
                    choices=[
                        (lora_name, str(i) ) for i, lora_name in enumerate(loras_names)
                    ],
                    value= default_loras_choices,
                    multiselect= True,
                    visible= len(loras)>0,
                    label="Activated Loras"
                )
                loras_mult_choices = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by space characters", value=default_loras_multis_str, visible= len(loras)>0 )

                show_advanced = gr.Checkbox(label="Show Advanced Options", value=False)
                with gr.Row(visible=False) as advanced_row:
                    with gr.Column():
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        guidance_scale = gr.Slider(1.0, 20.0, value=1.0, step=0.5, label="Guidance Scale")
                        flow_shift = gr.Slider(0.0, 25.0, value=7.0, step=0.1, label="Flow Shift") 
                        embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale")
                        repeat_generation = gr.Slider(1, 25.0, value=1.0, step=1, label="Number of Generated Video per prompt") 
                        tea_cache_setting = gr.Dropdown(
                            choices=[
                                ("Disabled", 0),
                                ("Fast (x1.6 speed up)", 0.1), 
                                ("Faster (x2.1 speed up)", 0.15), 
                            ],
                            value=0,
                            label="Tea Cache acceleration (the faster the acceleration the higher the degradation of the quality of the video)"
                        )

                show_advanced.change(fn=lambda x: gr.Row(visible=x), inputs=[show_advanced], outputs=[advanced_row])
            
            with gr.Column():
                gen_status = gr.Text(label="Status", interactive= False) 
                output = gr.Gallery(
                        label="Generated videos", show_label=False, elem_id="gallery"
                    , columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= False)
                state = gr.State({})
                generate_btn = gr.Button("Generate")
                abort_btn = gr.Button("Abort")

        gen_status.change(refresh_gallery, inputs = [state], outputs = output )

        abort_btn.click(abort_generation,state,abort_btn )
        output.select(select_video, state, None )

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
                embedded_guidance_scale,
                repeat_generation,
                tea_cache_setting,
                loras_choices,
                loras_mult_choices,
                state
            ],
            outputs= [gen_status] #,state 

        ).then( 
            finalize_gallery,
            [state], 
            [output , abort_btn]
        )
    
    return demo

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)

    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))

    server_name = args.server_name
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "0.0.0.0")

        
    demo = create_demo(args.model_base, args.save_path)
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)

    demo.launch(server_name=server_name, server_port=server_port)

 
