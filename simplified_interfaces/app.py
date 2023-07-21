import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import (
    Chat,
    Conversation,
    default_conversation,
    SeparatorStyle,
)
import decord

decord.bridge.set_bridge("torch")


from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--cfg-path",
        default="eval_configs/video_llama_eval.yaml",
        help="path to configuration file.",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print("Initializing Chat")
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)
chat = Chat(model, vis_processor, device="cuda:{}".format(args.gpu_id))
print("Initialization Finished")

# ========================================
#             Gradio Setting
# ========================================


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),
        gr.update(placeholder="Please upload your video first", interactive=False),
        gr.update(value="Upload & Start Chat", interactive=True),
        chat_state,
        img_list,
    )


def handle_describe_image(gr_img, text_input, chat_state, temperature, num_beams):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None

    chat_state = Conversation(
        system="You are able to understand the visual content that the user provides."
        "Follow the instructions carefully and explain your answers in detail.",
        roles=("Human", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="###",
    )
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)

    # gradio_ask
    if len(text_input) == 0:
        return (
            gr.update(interactive=True, placeholder="Input should not be empty!"),
            chat_state,
            None,
        )
    chat.ask(text_input, chat_state)

    # gradio_answer
    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=temperature,
        max_new_tokens=240,
        max_length=511,
    )[0]
    return llm_message, chat_state, img_list


def handle_describe_video(gr_video, text_input, chat_state, temperature, num_beams):
    if gr_video is None:
        return None, None, gr.update(interactive=True), chat_state, None

    chat_state = default_conversation.copy()
    chat_state = Conversation(
        system="You are able to understand the visual content that the user provides."
        "Follow the instructions carefully and explain your answers in detail.",
        roles=("Human", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="###",
    )
    img_list = []
    llm_message = chat.upload_video(gr_video, chat_state, img_list)

    # gradio_ask
    if len(text_input) == 0:
        return (
            gr.update(interactive=True, placeholder="Input should not be empty!"),
            chat_state,
            None,
        )
    chat.ask(text_input, chat_state)

    # gradio_answer
    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=temperature,
        max_new_tokens=240,
        max_length=511,
    )[0]
    return llm_message, chat_state, img_list


with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                video = gr.Video()
                describe_video_button = gr.Button(
                    value="Describe Video", interactive=True, variant="primary"
                )
            with gr.Column():
                image = gr.Image(type="filepath")
                describe_image_button = gr.Button(
                    value="Describe Image", interactive=True, variant="primary"
                )

        with gr.Row():
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            audio = gr.Checkbox(interactive=True, value=False, label="Audio")

        chat_state = gr.State()
        img_list = gr.State()

        text_input = gr.Textbox("describe the image", label="User", show_label=False)

        llama_output = gr.Textbox(
            lines=5, placeholder="Chat goes here...", readonly=True
        )

    describe_image_button.click(
        handle_describe_image,
        [image, text_input, chat_state, temperature, num_beams],
        [llama_output, chat_state, img_list],
    )

    describe_video_button.click(
        handle_describe_video,
        [video, text_input, chat_state, temperature, num_beams],
        [llama_output, chat_state, img_list],
    )

    demo.launch(share=False, enable_queue=True)
