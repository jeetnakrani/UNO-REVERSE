from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

# Silence TensorFlow warnings (if present)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_insights(prompt, max_length=300):
    inputs = pipe.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = pipe.model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    insight = pipe.tokenizer.batch_decode(outputs)[0]
    return insight.replace(prompt, "").strip()
