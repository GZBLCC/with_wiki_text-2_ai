import torch.nn as nn
from transformers import GPT2Config, GPT2Model
import torch
import json
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel,GPT2Tokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

original_vocab_size = 50257
new_vocab_size = len(tokenizer)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
vocab_size = len(tokenizer)

class simple_languageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=4, num_heads=4):
        super(simple_languageModel, self).__init__()

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=128,
            n_ctx=128,
        )
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_state = transformer_output.last_hidden_state
        logits = self.lm_head(hidden_state)
        return logits

hidden_dim = 512
num_layers = 4
num_heads = 4

model = simple_languageModel(
    vocab_size=original_vocab_size,
    hidden_size=512,
    num_layers=4,
    num_heads=4
).to(device)

model_path = "best_model.pth"
pretrained_state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(pretrained_state_dict)

model.transformer.resize_token_embeddings(new_vocab_size)
model.lm_head = nn.Linear(hidden_dim, new_vocab_size).to(device)

with torch.no_grad():
    original_wte_weight = model.transformer.wte.weight.data
    new_wte_weight = torch.zeros(new_vocab_size , original_wte_weight.size(1)).to(device)
    new_wte_weight[:original_vocab_size] = original_wte_weight
    new_wte_weight[original_vocab_size:] = torch.randn_like(new_wte_weight[original_vocab_size:])
    model.transformer.wte.weight.data = new_wte_weight

    original_lm_head_weight = model.lm_head.weight.data
    original_lm_head_bias = model.lm_head.bias.data
    new_lm_head_weight = torch.zeros(new_vocab_size, hidden_dim).to(device)
    new_lm_head_weight[:original_vocab_size] = original_lm_head_weight
    new_lm_head_weight[original_vocab_size:] = torch.randn_like(new_lm_head_weight[original_vocab_size:])
    model.lm_head.weight.data = new_lm_head_weight

    new_lm_head_bias = torch.zeros(new_vocab_size).to(device)
    new_lm_head_bias[:original_vocab_size] = original_lm_head_bias
    model.lm_head.bias.data = new_lm_head_bias



def generate_text_with_custom_vocab(model, prompt, max_length=50, device='cpu'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)],dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            temperature = 0.7
            next_token_logits = outputs[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return generated_text




input_text = "hello world"
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
input_tensor = torch.tensor([input_ids]).to(device)

attention_mask = (input_tensor != tokenizer.pad_token_id).long().to(device)

with torch.no_grad():
    output = model(input_tensor, attention_mask=attention_mask)

probs = torch.softmax(output, dim=-1)
predicted_ids = torch.argmax(probs, dim=-1)
predicted_text =  tokenizer.decode(predicted_ids[0].tolist(), skip_special_tokens=True)

generated = generate_text_with_custom_vocab(model, "hello world", device=device)
print("Generated text:", generated)

