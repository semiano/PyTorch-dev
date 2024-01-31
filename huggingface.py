from transformers import AutoModel, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import PyPDF2


print("!!! 0")
##Convert PDF to Text
pdf_file_path =  'The_Michael_Prophecies_Book_Two_1998_Volume_One_Initial_FInal.pdf'
txt_file_path = pdf_file_path.replace('.pdf', '.txt')

#with open(pdf_file_path, 'rb') as file:
#    reader = PyPDF2.PdfReader(file)
#    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
#        for page_num in range(len(reader.pages)):
#            page = reader.pages[page_num]
#            txt_file.write(page.extract_text())

print("!!! 100")         
# Load pre-trained model and tokenizer
model = AutoModel.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

print("!!! 200")
# Prepare dataset
dataset = TextDataset(
    tokenizer= tokenizer,
    file_path= txt_file_path,
    block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)

print("!!! 300") 
# Training arguments
training_args = TrainingArguments(
    output_dir="./model_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

print("!!! 400")
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("!!! 500")
# Fine-tuning
trainer.train()

print("!!! 600")
# Save the fine-tuned model
model.save_pretrained("./model_finetuned")
