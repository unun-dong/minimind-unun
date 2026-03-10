# MiniMind ChatGPT 简单问答
# 确保已经下载模型并安装依赖

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./MiniMind2"

print("加载模型...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model = model.eval().to(device)
    
    print("MiniMind ChatGPT 已启动（输入 exit 退出）")
    
    while True:
        question = input("\n你：")
        
        if question.lower() == "exit":
            break
        
        if not question.strip():
            continue
        
        # 构建正确的聊天格式
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question}
        ]
        
        # 使用 apply_chat_template 构建提示词
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码结果
        full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只取新生成的部分
        input_length = inputs.input_ids.shape[1]
        answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        print("AI：", answer)
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("再见！")
