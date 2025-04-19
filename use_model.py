import torch.nn as nn
from transformers import GPT2Config, GPT2Model
import torch
import json
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel,GPT2Tokenizer
import torch

# 设置设备为CUDA（如果可用）或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载GPT2的tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 原始词汇表大小
original_vocab_size = 50257
# 新的词汇表大小（包括特殊token）
new_vocab_size = len(tokenizer)
# 添加特殊token（padding token）
tokenizer.add_special_tokens({"pad_token":"<pad>"})
# 更新词汇表大小
vocab_size = len(tokenizer)

# 定义简单的语言模型类
class simple_languageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=4, num_heads=4):
        super(simple_languageModel, self).__init__()

        # 配置GPT2模型
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=128,
            n_ctx=128,
        )
        # 初始化GPT2模型
        self.transformer = GPT2Model(config)
        # 初始化线性层作为语言模型头
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # 前向传播
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask)
        # 获取最后一层的隐藏状态
        hidden_state = transformer_output.last_hidden_state
        # 计算logits
        logits = self.lm_head(hidden_state)
        return logits

# 设置模型参数
hidden_dim = 512
num_layers = 4
num_heads = 4

# 初始化模型并移动到指定设备
model = simple_languageModel(
    vocab_size=original_vocab_size,
    hidden_size=512,
    num_layers=4,
    num_heads=4
).to(device)

# 加载预训练模型参数
model_path = "best_model.pth"
pretrained_state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(pretrained_state_dict)

# 调整模型以适应新的词汇表大小
model.transformer.resize_token_embeddings(new_vocab_size)
model.lm_head = nn.Linear(hidden_dim, new_vocab_size).to(device)

# 在不计算梯度的情况下更新模型参数
with torch.no_grad():
    # 保存原始的词嵌入权重
    original_wte_weight = model.transformer.wte.weight.data
    # 初始化新的词嵌入权重
    new_wte_weight = torch.zeros(new_vocab_size , original_wte_weight.size(1)).to(device)
    # 复制原始词嵌入权重到新的词嵌入权重
    new_wte_weight[:original_vocab_size] = original_wte_weight
    # 随机初始化新的词嵌入权重
    new_wte_weight[original_vocab_size:] = torch.randn_like(new_wte_weight[original_vocab_size:])
    # 更新词嵌入权重
    model.transformer.wte.weight.data = new_wte_weight

    # 保存原始的语言模型头权重和偏置
    original_lm_head_weight = model.lm_head.weight.data
    original_lm_head_bias = model.lm_head.bias.data
    # 初始化新的语言模型头权重
    new_lm_head_weight = torch.zeros(new_vocab_size, hidden_dim).to(device)
    # 复制原始语言模型头权重到新的语言模型头权重
    new_lm_head_weight[:original_vocab_size] = original_lm_head_weight
    # 随机初始化新的语言模型头权重
    new_lm_head_weight[original_vocab_size:] = torch.randn_like(new_lm_head_weight[original_vocab_size:])
    # 更新语言模型头权重
    model.lm_head.weight.data = new_lm_head_weight

    # 初始化新的语言模型头偏置
    new_lm_head_bias = torch.zeros(new_vocab_size).to(device)
    # 复制原始语言模型头偏置到新的语言模型头偏置
    new_lm_head_bias[:original_vocab_size] = original_lm_head_bias
    # 更新语言模型头偏置
    model.lm_head.bias.data = new_lm_head_bias



# 定义生成文本的函数
def generate_text_with_custom_vocab(model, prompt, max_length=50, device='cpu'):
    model.eval()
    # 将输入文本编码为input_ids
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)],dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            outputs = model(input_ids)
            # 设置temperature
            temperature = 0.7
            # 计算下一个token的logits
            next_token_logits = outputs[:, -1, :] / temperature
            # 计算概率分布
            probs = torch.softmax(next_token_logits, dim=-1)
            # 根据概率分布采样下一个token
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 如果遇到结束token则停止生成
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            # 更新input_ids
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return generated_text




# 输入文本
input_text = "hello world"
# 将输入文本编码为input_ids
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
# 将input_ids转换为tensor并移动到指定设备
input_tensor = torch.tensor([input_ids]).to(device)

# 创建attention mask
attention_mask = (input_tensor != tokenizer.pad_token_id).long().to(device)

# 在不计算梯度的情况下进行前向传播
with torch.no_grad():
    output = model(input_tensor, attention_mask=attention_mask)

# 计算概率分布
probs = torch.softmax(output, dim=-1)
# 获取预测的token ids
predicted_ids = torch.argmax(probs, dim=-1)
# 解码预测的文本
predicted_text =  tokenizer.decode(predicted_ids[0].tolist(), skip_special_tokens=True)

# 使用自定义词汇表生成文本
generated = generate_text_with_custom_vocab(model, "hello world", device=device)
print("Generated text:", generated)

