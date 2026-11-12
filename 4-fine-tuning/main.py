from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


input_file = "sherlock.txt"              # Input file with text for training. 
trained_model_dir = "trained_model"      # Directory where the trained model will be saved. 
trained_model_name = "fine_tuned_model"  # Name of the trained model (to be used in Ollama).


# ==============================================================================
# 1. CONFIGURATION AND LOADING THE BASE MODEL IN 4-BIT FORMAT
# ==============================================================================

# Define the maximum sequence (context) length in tokens.
# Llama 3.2 models natively support long contexts, but to save VRAM
# during training, it is worth limiting it to e.g. 2048 or 4096.
max_seq_length = 2048 

# The Unsloth library automatically downloads optimized models from Hugging Face.
# We use a pre-quantized (4-bit) version, which allows fine-tuning models
# like Llama 3.2 (3 billion parameters) on consumer GPUs (e.g. with 12-16GB VRAM).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.2-3b-Instruct-bnb-4bit", # Name of the optimized base model
    max_seq_length = max_seq_length,                       # Pass the text length limit
    load_in_4bit = True,                                   # Enable 4-bit compression (memory saving)
    # device_map = "auto"                                  # Optional: automatic mapping to available GPUs
)

# ==============================================================================
# 2. PREPARING THE MODEL FOR LoRA / QLoRA (TRAINING PARAMETERS)
# ==============================================================================

# Instead of training billions of base weights (which would take hundreds of gigabytes of memory),
# we freeze the main model and attach small, trainable adapter layers to it.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,               # Rank (r): determines the size/width of LoRA layers. A higher value (e.g. 32, 64) 
                          # allows the model to learn more complex patterns, but consumes more VRAM.
    lora_alpha = 16,      # Scaling factor for LoRA. Usually set to the same value as 'r' or twice 'r'.
    lora_dropout = 0,     # Dropout rate. For Unsloth, the optimal value is 0 (speeds up training).
    bias = "none",        # Bias configuration. "none" is recommended for stable and fast QLoRA training.
    
    # Target Modules: Specify exactly which internal network layers we want to fine-tune.
    # Targeting all main Attention modules and MLP networks (q_proj, k_proj, etc.)
    # yields the highest fine-tuning quality and better adaptation to new style/knowledge.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)



# ==============================================================================
# 3. DATA PREPARATION FROM TEXT FILE AND TRAINING PROCESS
# ==============================================================================

# 3.1. Read the text file into memory
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 3.2. Split the text into smaller chunks (e.g., paragraphs)
# Language models cannot process a huge file at once. We split it by double newlines.
chunks = [para.strip() for para in raw_text.split("\n\n") if len(para.strip()) > 10]

# 3.3. Convert to a format accepted by Hugging Face
# Create a dictionary with the key "text", containing the list of our text chunks.
data_dict = {"text": chunks}
dataset = Dataset.from_dict(data_dict)

# 3.4. Define the Prompt Template for the instruction model
# Since we are using an "Instruct" model (Llama 3.2 Instruct), we need to wrap our text
# in appropriate system tags so the model learns within the proper context.
def format_prompts(examples):
    formatted_texts = []
    for text in examples["text"]:
        # Use the official Llama 3 format (<|begin_of_text|>, <|start_header_id|>, etc. tags)
        # Provide the text as system information / base knowledge for the model to memorize.
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "Memorize the following information and learn it by heart:\n"
            f"{text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "Confirm that you understood.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            "I understood and saved this information in my memory.<|eot_id|>"
        )
        formatted_texts.append(prompt)
    return {"formatted_text": formatted_texts}

# Map (transform) our dataset using the function above
dataset = dataset.map(format_prompts, batched=True)

# 3.5. Configure and initialize the trainer (SFTTrainer)
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "formatted_text",
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 5,
        output_dir = "outputs",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
    ),
)


# Start the actual fine-tuning process
print("Starting fine-tuning on the text file...")
trainer.train()
print("Fine-tuning completed.")


# ==============================================================================
# 4. EXPORT AND SAVE TO GGUF FORMAT (FOR OLLAMA)
# ==============================================================================

# After training finishes, Unsloth allows exporting the merged model
# (base weights + trained LoRA layers) directly into a single .gguf file.
# This file will be saved on your disk in the folder defined in "trained_model_dir".
model.save_pretrained_gguf(
    trained_model_dir,               # Output directory name
    tokenizer,                       # Tokenizer matched to the model architecture
    quantization_method = "q4_k_m"   # Quantization method (q4_k_m is the sweet spot between file size, speed, and preserving model intelligence)
)

print(f"Create the model in Ollama. Enter in the terminal:")
print(f"ollama create {trained_model_name} -f ./{trained_model_dir}/Modelfile")
print()
print(f"Check the list of models:")
print(f"ollama list")
print()
print(f"Run the model in Ollama:")
print(f"ollama run {trained_model_name}")
print()

