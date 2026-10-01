from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")

# Store conversation history
history = [{"role": "system", "content": "You are a helpful, concise assistant."}]

print("Chatbot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"quit", "exit"}:
        break

    if not user_input:
        continue

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # Generate chatbot response
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Save and display response
    history.append({"role": "assistant", "content": reply})
    print(f"Bot: {reply}\n")
