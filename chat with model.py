import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for computations.")
else:
    print("CUDA is not available. Using CPU.")
print(torch.version.cuda)
    
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-6b")

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to('cuda')

def chat(user_input):
    # Encode user input, add end of string token, and move to GPU
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    if torch.cuda.is_available():
        new_user_input_ids = new_user_input_ids.to('cuda')

    # Generate a response
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Move back to CPU for decoding
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Example usage
user_input = "Hello, how are you?"
print("User:", user_input)
print("Bot:", chat(user_input))
