from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import pdb

prompt = "Alice and Bob"
checkpoint = "EleutherAI/pythia-1.4b-deduped"
assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
print(sum(p.numel() for p in model.parameters()))
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
print(sum(p.numel() for p in assistant_model.parameters()))

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")
pdb.set_trace()
outputs = model.generate(**inputs, assistant_model=assistant_model, pad_token_id=tokenizer.eos_token_id)
o=tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(o)
