# Fine-Tune-Multimodal-LLM-Idefics-2-using-QLoRA

## Overview
This project demonstrates fine-tuning the multimodal large language model (LLM) "Idefics 2" using Quantized Low-Rank Adaptation (QLoRA). The approach optimizes GPU memory usage and facilitates efficient training while maintaining the model's performance across text and image modalities.

---

## Key Features

1. **Model**: Idefics 2, a multimodal LLM capable of processing text and visual data.
2. **Fine-Tuning Method**: QLoRA for efficient low-memory adaptation of large models.
3. **Optimization Techniques**:
   - 4-bit quantization of model weights.
   - LoRA (Low-Rank Adaptation) for task-specific fine-tuning.
   - Freezing pre-trained layers to reduce computation.
4. **Applications**: Tasks like visual question answering, image captioning, and multimodal reasoning.

---

## Installation

Install the required dependencies:
```bash
pip install transformers accelerate datasets bitsandbytes peft
```

---

## Steps for Fine-Tuning

### 1. Model Setup
Load the pre-trained Idefics 2 model with 4-bit quantization:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Idefics-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Enable 4-bit quantization
    device_map="auto"
)
```

### 2. Apply QLoRA
Add LoRA adapters to the model:
```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Rank of the low-rank matrices
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
```

### 3. Training Configuration
Set up the training loop using `Trainer`:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
```

---

- **Task**: Fine-tuning the Idefics 2 model to handle multimodal tasks efficiently.
- **Challenge**: High memory requirements and computational overhead associated with large LLMs.
- Reduce the GPU memory footprint while maintaining performance for multimodal tasks.
- Implement QLoRA for low-resource fine-tuning.

### Action
1. Quantized the model weights to 4-bit precision to reduce memory usage.
2. Used LoRA adapters to introduce low-rank updates, focusing on task-specific layers.
3. Leveraged gradient accumulation and small batch sizes to optimize GPU usage.
4. Configured efficient training arguments and evaluation metrics.
5. Monitored performance trade-offs to balance memory efficiency and accuracy.

### Result
- Achieved successful fine-tuning of Idefics 2 with significantly reduced GPU memory requirements.
- Maintained high performance on multimodal tasks like visual question answering and image captioning.

---

## Evaluation Metrics

1. **ANLS**: Average Normalized Levenshtein Similarity, for OCR or text-based tasks.
2. **Image-Text Alignment**: Metrics such as BLEU, CIDEr, or CLIP similarity.
3. **Performance Monitoring**: Check trade-offs in memory efficiency and task accuracy.

---

## Benefits of Using QLoRA

- **Reduced Memory Footprint**: Fine-tune large models on consumer-grade GPUs.
- **Task-Specific Adaptation**: Focus on fine-tuning only task-relevant weights.
- **Multimodal Flexibility**: Handle complex multimodal tasks efficiently.

---

## Future Work

- Experiment with larger datasets for improved generalization.
- Extend to other multimodal tasks like video understanding.
- Incorporate advanced evaluation techniques for better benchmarking.

---

## References

- Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- PEFT Library: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
