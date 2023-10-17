from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

def preprocess_function(example):
    return tokenizer(f"Original email:\n{example['simple_mail']}\nFixed email:\n{example['email']}</s>", truncation=True)


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokenizer.model_max_length = 1024

dataset = load_dataset("csv", data_files="data/simple_mail.csv", split='train')
dataset = dataset.map(
    preprocess_function,
    num_proc=6,
    remove_columns=dataset.column_names,
)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="mail_fixer",
    save_strategy='epoch',
    num_train_epochs=2,
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=16,
    push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()
