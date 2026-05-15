import torch 
import tiktoken 
from mini_GPT import GPTModel
import argparse


def generate_text(model, input_ids, max_new_tokens, context_length, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        input_context = input_ids[:, -context_length:]
        
        with torch.no_grad():
            logits = model(input_context)
        
        last_token_logits = logits[:, -1, :]
        
        # Temperature scaling
        last_token_logits = last_token_logits / temperature
        
        if top_k is not None:
            top_values, _ = torch.topk(last_token_logits, top_k)
            
            min_top_value =  top_values[:, -1].unsqueeze(-1)
            
            last_token_logits = torch.where(
                last_token_logits < min_top_value,
                torch.tensor(float("-inf")).to(last_token_logits.device),
                last_token_logits
            )
        
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
        
    return input_ids


GPT_CONFIG = {
    "vocab_size" : 50257,
    "context_length" : 128, 
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

device = "cuda" if torch.cuda.is_available() else "cpu" 

tokenizer = tiktoken.get_encoding("gpt2") 


model = GPTModel(GPT_CONFIG)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.to(device)
model.eval()


# prompt = "How is your day going ? " 

# input_ids = tokenizer.encode(prompt)

# input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

# output_ids = generate_text(
#     model=model,
#     input_ids=input_ids,
#     max_new_tokens=50,
#     context_length=GPT_CONFIG["context_length"]
# )

# output_text = tokenizer.decode(output_ids[0].tolist())

# print(output_text)


parser = argparse.ArgumentParser()

parser.add_argument("--prompt", type=str, default="How is your day going?")
parser.add_argument("--max_new_tokens", type=int, default=80)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=50)

args = parser.parse_args()

input_ids = tokenizer.encode(args.prompt)
input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

output_ids = generate_text(
    model=model,
    input_ids=input_ids,
    max_new_tokens=args.max_new_tokens,
    context_length=GPT_CONFIG["context_length"],
    temperature=args.temperature,
    top_k=args.top_k
)

output_text = tokenizer.decode(output_ids[0].tolist())

print(output_text)


# CUDA_VISIBLE_DEVICES=0 python generate.py   --prompt "The artist looked"   --max_new_tokens 100