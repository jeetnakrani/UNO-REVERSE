
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_insights(prompt, max_length=250):
    messages = [
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)

    outputs = pipe(
        formatted_prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    insight = outputs[0]['generated_text']
    return insight.split("[/INST]")[-1].strip()

