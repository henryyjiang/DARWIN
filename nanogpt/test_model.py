from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("codegen-2B")
model = AutoModelForCausalLM.from_pretrained("codegen-2B")


prompt = """

# Python function to compute Fibonacci numbers
def fibonacci(n):
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.2,
    do_sample=True,
    top_p=0.95,
)

generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
