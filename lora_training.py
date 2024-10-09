from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import load_dataset
import torch

def main():

    # LLaMA the pre-trained LLaMA model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # add pad token because Llama does not define it
    tokenizer.pad_token = tokenizer.eos_token

    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Parts of the model to apply LoRA to
        lora_dropout=0.1,  # Dropout to avoid overfitting
        bias="none"  # No bias parameters to be modified
    )

    # Apply the LoRA configuration to the model
    model = get_peft_model(model, lora_config)

    # Load the dataset
    dataset_path = "chatbot_dataset.csv"
    # dataset = pd.read_csv(dataset_path, names=["prompt", "response"])
    dataset = load_dataset('csv', data_files=dataset_path)

    # Tokenize the prompts and responses
    def tokenize_function(examples):
        return tokenizer(
                examples["prompt"],
                examples["response"],
                truncation=True,
                padding="max_length"
            )

    # tokenized_dataset = dataset.apply(tokenize_function, axis=1)
    tokenized_dataset = dataset.map(tokenize_function)

    # Create a dataset for training and data collator for batching
    train_dataset = tokenized_dataset["train"]
    # Create the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,  # Try reducing this value
        per_device_eval_batch_size=1,   # Same for evaluation
        gradient_accumulation_steps=8,   # Simulate a batch size of 8
        fp16=True,  # Enable mixed precision training for lower memory usage
        output_dir="lora_model",
        num_train_epochs=3,  # Adjust the number of epochs as needed
        save_steps=1000,
        evaluation_strategy="steps",
        logging_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Clears the cached memory
    torch.cuda.empty_cache()

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model("./fine-tuned-llama-lora")


if __name__ == "__main__":
    main()