from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel,GPT2Tokenizer
import torch.optim as optim
import math
from torch.nn.utils.rnn import pad_sequence
import sys
import psutil
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import torch.cuda


class simple_languageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers = 4, num_heads = 4):
        super(simple_languageModel, self).__init__()
        
        config = GPT2Config(
            vocab_size = vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=128,
            n_ctx=128,
        )
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask = None):
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_state = transformer_output.last_hidden_state
        logits = self.lm_head(hidden_state)
        return logits




def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return tokenized

def collate_fn(batch, tokenizer):
    if len(batch) == 0:
        return {
            'input_ids': torch.tensor([]),
            'attention_mask': torch.tensor([])
        }

    input_ids = [item['input_ids']for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}



def generate_text(model, prompt,tokenizer,max_length=50,device = 'cpu'):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)


    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            temperature = 0.7
            next_token_logits = outputs[:, -1, :] / temperature
            next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    generate_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generate_text

def train_model(model,train_loader,val_loader,loss_fn,optimizer,scheduler,save_path,vocab_size,device,grad_accum_step,num_epoch = 10, patience = 3):
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler()
    grad_accum_steps = grad_accum_step
    torch.cuda.empty_cache()

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        train_loader_with_progress = tqdm(train_loader,desc=f"Epoch{epoch+1}/{num_epoch}",leave = False)
        for step, batch in enumerate(train_loader_with_progress):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = input_ids.detach().clone().to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(input_ids,attention_mask = attention_mask)
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
                loss = (loss * attention_mask.view(-1)).mean() / grad_accum_steps

            scaler.scale(loss).backward()
            if (step+1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            train_loader_with_progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')
        val_loss = evaluate_model(model=model,
                                val_loader=val_loader,
                                loss_fn=loss_fn,
                                vocab_size=vocab_size,
                                device=device)
        if scheduler:
            scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current Learning Rate: {current_lr}')

def evaluate_model(model, val_loader,loss_fn,vocab_size,device):
    model.eval()
    total_loss = 0
    torch.cuda.empty_cache()

    val_loader_with_process = tqdm(val_loader,desc="Validation",leave = False)
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device,non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = input_ids.clone().detach().to(device, non_blocking=True)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            loss = (loss * attention_mask.view(-1)).mean()
            total_loss += loss.item()

            val_loader_with_process.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(val_loader)
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float('inf')

        print(f'Validation Loss: {avg_loss}')
        print(f'困惑度:{perplexity}')
        return avg_loss

def print_system_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_info.percent}%")

def main():
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad Token ID: {tokenizer.pad_token_id}, Text: {tokenizer.decode([tokenizer.pad_token_id])}")
    vocab_size = tokenizer.vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = simple_languageModel(vocab_size=vocab_size,
                                hidden_size=512,
                                num_layers=4,
                                num_heads=4
                                ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(reduce='none')

    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_datasets.set_format(type = 'torch', columns=['input_ids', 'attention_mask'])

    train_loader = DataLoader(
                            tokenized_datasets['train'],
                            batch_size=8,
                            shuffle=True,
                            collate_fn=lambda x: collate_fn(x, tokenizer),
                            num_workers=0,
                            pin_memory=True)
    val_loader = DataLoader(tokenized_datasets['validation'],
                            batch_size=32,
                            shuffle=False,
                            collate_fn=lambda x: collate_fn(x, tokenizer),num_workers=0)
    num_epoch = 10

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min',factor = 0.1,patience = 3)
    train_model(
        model=model,
        grad_accum_step = 4,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        vocab_size=vocab_size,
        device=device,
        num_epoch=num_epoch,
        patience=5,
        save_path='best_model.pth',
        scheduler = scheduler
    )
    print(f"Pad Token ID: {tokenizer.pad_token_id}, Text: {tokenizer.decode([tokenizer.pad_token_id])}")
    generated_text = generate_text(model, "Once upon a time", tokenizer, max_length=50, device=device)
    print(generated_text)
if __name__ == '__main__':
    main()



