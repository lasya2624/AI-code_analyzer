
# Install necessary packages
!pip install transformers datasets torch protobuf sentencepiece --quiet

# All imports
import json
import random
import os
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# ✅ Generate 1000+ synthetic code samples
function_templates = [
    {
        "code": "def add(a, b): return a + b",
        "explanation": "This function adds two numbers and returns their sum.",
        "documentation": "### Description:\nAdds two numbers.\n\n### Args:\n- **a (int)**: The first number.\n- **b (int)**: The second number.\n\n### Returns:\n- **int**: The sum of `a` and `b`."
    },
    {
        "code": "def multiply(a, b): return a * b",
        "explanation": "This function multiplies two numbers and returns the result.",
        "documentation": "### Description:\nMultiplies two numbers.\n\n### Args:\n- **a (int)**: First number.\n- **b (int)**: Second number.\n\n### Returns:\n- **int**: The product of `a` and `b`."
    },
    {
        "code": "def factorial(n): return 1 if n == 0 else n * factorial(n - 1)",
        "explanation": "This function calculates the factorial of a number using recursion.",
        "documentation": "### Description:\nComputes the factorial of a number.\n\n### Args:\n- **n (int)**: The number to calculate factorial for.\n\n### Returns:\n- **int**: The factorial of `n`."
    },
    {
        "code": "def reverse_string(s): return s[::-1]",
        "explanation": "This function reverses a given string.",
        "documentation": "### Description:\nReverses a string.\n\n### Args:\n- **s (str)**: The string to reverse.\n\n### Returns:\n- **str**: The reversed string."
    },
    {
        "code": "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
        "explanation": "This function checks if a number is prime.",
        "documentation": "### Description:\nChecks if a number is prime.\n\n### Args:\n- **n (int)**: The number to check.\n\n### Returns:\n- **bool**: `True` if the number is prime, `False` otherwise."
    },
    {
        "code": "def fibonacci(n): return n if n <= 1 else fibonacci(n - 1) + fibonacci(n - 2)",
        "explanation": "This function calculates the nth Fibonacci number using recursion.",
        "documentation": "### Description:\nComputes the nth Fibonacci number recursively.\n\n### Args:\n- **n (int)**: The position in the Fibonacci sequence.\n\n### Returns:\n- **int**: The nth Fibonacci number."
    }
]

dataset = [random.choice(function_templates) for _ in range(1000)]
df = pd.DataFrame(dataset)

# ✅ Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df)

# ✅ Load model & tokenizer
model_name = "Salesforce/codet5-small"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ✅ Preprocessing
def preprocess_function(examples):
    inputs = ["explain and document: " + code for code in examples["code"]]
    outputs = [
        "### Explanation:\n" + e + "\n\n### Documentation:\n" + d
        for e, d in zip(examples["explanation"], examples["documentation"])
    ]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ✅ Tokenize and split dataset
tokenized = hf_dataset.map(preprocess_function, batched=True)
split_data = tokenized.train_test_split(test_size=0.2)

# ✅ Set training arguments
os.environ["WANDB_DISABLED"] = "true"
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=10,
    fp16=torch.cuda.is_available()
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_data["train"],
    eval_dataset=split_data["test"]
)

# ✅ Train the model
trainer.train()

# ✅ Save model and tokenizer
model.save_pretrained("code_explainer_model")
tokenizer.save_pretrained("code_explainer_model")

# ✅ Test function
def generate_explanation_and_documentation(code_snippet):
    inputs = tokenizer("explain and document: " + code_snippet, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Example usage
test_code = "def multiply(a, b): return a * b"
generate_explanation_and_documentation(test_code)

