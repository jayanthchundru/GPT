import torch
from dataset import create_dataloader
import tiktoken
from mini_GPT import GPTModel
import torch.nn as nn


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()
    

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = tokenizer.encode(text)

split_idx = int(0.9 * len(token_ids))
train_ids = token_ids[:split_idx]
val_ids = token_ids[split_idx:]

context_length = 128
batch_size = 4

train_loader = create_dataloader(
    token_ids=train_ids,
    context_length=context_length,
    batch_size=batch_size,
    shuffle=True
)

val_loader = create_dataloader(
    token_ids=val_ids,
    context_length=context_length,
    batch_size=batch_size,
    shuffle=False
)

device="cuda" if torch.cuda.is_available() else "cpu"

GPT_CONFIG = {
    "vocab_size" : 50257,
    "context_length" : context_length, 
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

model = GPTModel(GPT_CONFIG)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-4
)

def eval_loss(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0 
    num_batches = 0
    
    with torch.no_grad():
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            
            loss = criterion(logits, y)
            
            total_loss += loss
            num_batches += 1
    
    model.train()

    return total_loss / num_batches

best_val_loss = float("inf")

for epoch in range(3):
    for step, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        
        # print("Original logits shape:", logits.shape)
        
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        
        # print("Flattened logits shape: ", logits.shape)
        # print("Flattened targets shape: ", y.shape)
        
        # compute the loss
        loss = criterion(logits, y)
        
        #clear the old gradients
        optimizer.zero_grad()
        
        # compute gradients
        loss.backward()
        
        # Update Parameters
        optimizer.step()
        
        if step % 100  == 0:
            val_loss = eval_loss(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}, Val Loss: {val_loss}")
    
    #saves the best model based on val loss upon each epoch
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(model.state_dict(), "checkpoints/best_model.pt")

        print(f"Saved best model | Best Val Loss: {best_val_loss:.4f}")