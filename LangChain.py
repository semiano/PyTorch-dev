import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for computations.")
else:
    print("CUDA is not available. Using CPU.")
print(torch.version.cuda)


# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
if torch.cuda.is_available(): next
    #model = model.to('cuda')
    
# Set the model to evaluation mode
model.eval()

def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=50256)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

print("Chatbot: Hi there! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = generate_response(user_input)
    print("Chatbot:", response)
