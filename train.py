from datasets import load_dataset  # 导入数据集加载库
from torch.utils.data import DataLoader  # 导入数据加载器
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel,GPT2Tokenizer  # 导入GPT2相关模块
import torch.optim as optim  # 导入PyTorch优化器
import math  # 导入数学库
from torch.nn.utils.rnn import pad_sequence  # 导入填充序列函数
import sys  # 导入系统库
import psutil  # 导入系统利用率库
from tqdm import tqdm  # 导入进度条库
from torch.optim.lr_scheduler import OneCycleLR  # 导入学习率调度器
from torch.cuda.amp import autocast, GradScaler  # 导入自动混合精度训练相关模块
import torch.cuda  # 导入PyTorch CUDA模块


# 定义简单的语言模型类
class simple_languageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers = 4, num_heads = 4):
        super(simple_languageModel, self).__init__()  # 调用父类构造函数
        
        # 配置GPT2模型
        config = GPT2Config(
            vocab_size = vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=128,
            n_ctx=128,
        )
        self.transformer = GPT2Model(config)  # 初始化GPT2模型
        self.lm_head = nn.Linear(hidden_size, vocab_size)  # 初始化线性层作为语言模型头部

    def forward(self, input_ids, attention_mask = None):
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask)  # 前向传播
        hidden_state = transformer_output.last_hidden_state  # 获取最后隐藏状态
        logits = self.lm_head(hidden_state)  # 计算logits
        return logits  # 返回logits




# 定义文本标记化函数
def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)  # 标记化文本
    return tokenized  # 返回标记化结果

# 定义批处理函数
def collate_fn(batch, tokenizer):
    if len(batch) == 0:
        return {
            'input_ids': torch.tensor([]),
            'attention_mask': torch.tensor([])
        }

    input_ids = [item['input_ids']for item in batch]  # 获取输入ID
    attention_mask = [item['attention_mask'] for item in batch]  # 获取注意力掩码

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)  # 填充输入ID
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)  # 填充注意力掩码

    return {'input_ids': input_ids, 'attention_mask': attention_mask}  # 返回批处理结果



# 定义文本生成函数
def generate_text(model, prompt,tokenizer,max_length=50,device = 'cpu'):
    model.eval()  # 设置模型为评估模式
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)  # 标记化提示文本


    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)  # 模型前向传播
            temperature = 0.7  # 温度参数
            next_token_logits = outputs[:, -1, :] / temperature  # 计算下一个token的logits
            next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)  # 采样下一个token

            if next_token_id.item() == tokenizer.eos_token_id:  # 如果是结束符则停止
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=1)  # 更新输入ID

    generate_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)  # 解码生成文本
    return generate_text  # 返回生成文本

# 定义模型训练函数
def train_model(model,train_loader,val_loader,loss_fn,optimizer,scheduler,save_path,vocab_size,device,grad_accum_step,num_epoch = 10, patience = 3):
    best_val_loss = float('inf')  # 初始化最佳验证损失
    patience_counter = 0  # 初始化耐心计数器
    scaler = torch.cuda.amp.GradScaler()  # 初始化梯度缩放器
    grad_accum_steps = grad_accum_step  # 设置梯度累积步数
    torch.cuda.empty_cache()  # 清空CUDA缓存

    for epoch in range(num_epoch):
        model.train()  # 设置模型为训练模式
        total_loss = 0  # 初始化总损失
        train_loader_with_progress = tqdm(train_loader,desc=f"Epoch{epoch+1}/{num_epoch}",leave = False)  # 进度条
        for step, batch in enumerate(train_loader_with_progress):
            input_ids = batch['input_ids'].to(device, non_blocking=True)  # 获取输入ID
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)  # 获取注意力掩码
            labels = input_ids.detach().clone().to(device, non_blocking=True)  # 获取标签
            with torch.cuda.amp.autocast():
                logits = model(input_ids,attention_mask = attention_mask)  # 模型前向传播
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))  # 计算损失
                loss = (loss * attention_mask.view(-1)).mean() / grad_accum_steps  # 计算平均损失

            scaler.scale(loss).backward()  # 反向传播
            if (step+1) % grad_accum_steps == 0:
                scaler.step(optimizer)  # 更新优化器
                scaler.update()  # 更新梯度缩放器
                optimizer.zero_grad()  # 清空梯度

            total_loss += loss.item() * grad_accum_steps  # 累加损失
            train_loader_with_progress.set_postfix(loss=loss.item())  # 更新进度条

        avg_train_loss = total_loss / len(train_loader)  # 计算平均训练损失
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')  # 打印损失
        val_loss = evaluate_model(model=model, val_loader=val_loader, loss_fn=loss_fn, vocab_size=vocab_size, device=device,
                                val_loader=val_loader,
                                loss_fn=loss_fn,
                                vocab_size=vocab_size,
                                device=device)  # 验证模型
        # 如果scheduler存在，则执行scheduler的step方法，传入验证损失值
        if scheduler:
            scheduler.step(val_loss)
        # 如果当前验证损失小于最佳验证损失，则更新最佳验证损失，重置耐心计数器，并保存模型状态
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            # 否则，增加耐心计数器，如果耐心计数器达到设定的耐心值，则提前停止训练
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break
        # 获取当前学习率并打印
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current Learning Rate: {current_lr}')

def evaluate_model(model, val_loader,loss_fn,vocab_size,device):
    # 设置模型为评估模式
    model.eval()
    total_loss = 0
    # 清空CUDA缓存
    torch.cuda.empty_cache()

    # 使用tqdm包装验证数据加载器，以便在进度条中显示验证进度
    val_loader_with_process = tqdm(val_loader,desc="Validation",leave = False)
    # 在不计算梯度的情况下进行评估
    with torch.no_grad():
        for batch in val_loader:
            # 将输入数据移动到指定设备
            input_ids = batch['input_ids'].to(device,non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = input_ids.clone().detach().to(device, non_blocking=True)

            # 获取模型输出
            logits = model(input_ids, attention_mask=attention_mask)
            # 计算损失
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            # 根据注意力掩码计算加权平均损失
            loss = (loss * attention_mask.view(-1)).mean()
            total_loss += loss.item()

            # 更新进度条的损失值
            val_loader_with_process.set_postfix(loss=loss.item())

        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        # 计算困惑度，如果损失值过大导致溢出，则设为无穷大
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float('inf')

        # 打印验证损失和困惑度
        print(f'Validation Loss: {avg_loss}')
        print(f'困惑度:{perplexity}')
        return avg_loss

def print_system_usage():
    # 获取CPU使用率
    cpu_usage = psutil.cpu_percent(interval=1)
    # 获取内存使用信息
    memory_info = psutil.virtual_memory()
    # 打印CPU和内存使用情况
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_info.percent}%")

def main():
    # 加载数据集
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    # 加载预训练的GPT2分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # 设置填充标记
    tokenizer.pad_token = tokenizer.eos_token
    # 打印填充标记的ID和对应的文本
    print(f"Pad Token ID: {tokenizer.pad_token_id}, Text: {tokenizer.decode([tokenizer.pad_token_id])}")
    # 获取词汇表大小
    vocab_size = tokenizer.vocab_size
    # 设置设备为CUDA（如果可用）或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化简单的语言模型，并移动到指定设备
    model = simple_languageModel(vocab_size=vocab_size,
                                hidden_size=512,
                                num_layers=4,
                                num_heads=4
                                ).to(device)
    
    # 初始化优化器，使用AdamW算法，学习率为1e-4，权重衰减为1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(reduce='none')

    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    # 设置数据集格式
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



