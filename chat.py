from airllm import AutoModel
from transformers import AutoTokenizer
import config

print("🔄 Carregando modelo... (primeira vez pode demorar)\n")

model = AutoModel.from_pretrained(
    config.MODEL_ID,
    compression=config.COMPRESSION,
    offload_folder=config.OFFLOAD_FOLDER,
)

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

print("✅ Modelo carregado! Digite 'sair' para encerrar.\n")

# Histórico de mensagens
history = []

while True:
    user_input = input("Você: ").strip()

    if user_input.lower() in ("sair", "exit", "quit"):
        print("Encerrando...")
        break

    if not user_input:
        continue

    # Adiciona ao histórico
    history.append({"role": "user", "content": user_input})

    # Monta o prompt com histórico
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_LENGTH,
    )

    print("\nAssistente: ", end="", flush=True)

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        do_sample=True,
        use_cache=True,
    )

    # Decodifica só a parte nova
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(response)
    print()

    # Adiciona resposta ao histórico
    history.append({"role": "assistant", "content": response})
