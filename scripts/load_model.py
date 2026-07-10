from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "CohereForAI/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
