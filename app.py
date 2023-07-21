import gradio as gr
from share_btn import community_icon_html, loading_icon_html, share_js
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

def infer(prompt):
    #prompt = "Darth Vader is surfing on waves"
    video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    video_path = export_to_video(video_frames)
    print(video_path)
    return video_path, gr.Group.update(visible=True)

css = """
#col-container {max-width: 510px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
      transform: rotate(0deg);
  }
  to {
      transform: rotate(360deg);
  }
}

#share-btn-container {
  display: flex; 
  padding-left: 0.5rem !important; 
  padding-right: 0.5rem !important; 
  background-color: #000000; 
  justify-content: center; 
  align-items: center; 
  border-radius: 9999px !important; 
  max-width: 13rem;
}

#share-btn-container:hover {
  background-color: #060606;
}

#share-btn {
  all: initial; 
  color: #ffffff;
  font-weight: 600; 
  cursor:pointer; 
  font-family: 'IBM Plex Sans', sans-serif; 
  margin-left: 0.5rem !important; 
  padding-top: 0.5rem !important; 
  padding-bottom: 0.5rem !important;
  right:0;
}

#share-btn * {
  all: unset;
}

#share-btn-container div:nth-child(-n+2){
  width: auto !important;
  min-height: 0px !important;
}

#share-btn-container .wrap {
  display: none !important;
}

#share-btn-container.hidden {
  display: none!important;
}
img[src*='#center'] { 
    display: block;
    margin: auto;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center;">Zeroscope Text-to-Video</h1>
            <p style="text-align: center;">
            A watermark-free Modelscope-based video model optimized for producing high-quality 16:9 compositions and a smooth video output. <br />
            </p>
            
            [![Duplicate this Space](https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm.svg#center)](https://huggingface.co/spaces/fffiloni/zeroscope?duplicate=true)
            
            """
        )

        prompt_in = gr.Textbox(label="Prompt", placeholder="Darth Vader is surfing on waves", elem_id="prompt-in")
        #inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=40, interactive=False)
        submit_btn = gr.Button("Submit")
        video_result = gr.Video(label="Video Output", elem_id="video-output")

        with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
            community_icon = gr.HTML(community_icon_html)
            loading_icon = gr.HTML(loading_icon_html)
            share_button = gr.Button("Share to community", elem_id="share-btn")

    submit_btn.click(fn=infer,
                    inputs=[prompt_in],
                    outputs=[video_result, share_group])
    
    share_button.click(None, [], [], _js=share_js)

demo.queue(max_size=12).launch(share=True)
        
