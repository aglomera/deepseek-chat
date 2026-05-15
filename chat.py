"""
╔══════════════════════════════════════════════════════════╗
║           AirLLM - Script de Início Rápido               ║
║   Roda LLMs de 70B parâmetros com apenas 4GB de VRAM     ║
╚══════════════════════════════════════════════════════════╝

TUTORIAL RÁPIDO:
────────────────
1. Instale as dependências:
   pip install airllm bitsandbytes

2. (Opcional) Para modelos fechados como Llama 3, autentique no HuggingFace:
   huggingface-cli login

3. Execute este script:
   python airllm_starter.py

NOTAS:
  - Na 1ª execução, o modelo é baixado e fragmentado em camadas (pode demorar)
  - Garanta espaço em disco suficiente (~140GB para modelos 70B)
  - Escolha o MODO desejado ajustando a variável MODE abaixo
"""

import sys
import torch
from airllm import AutoModel

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO — ajuste aqui conforme sua GPU
# ─────────────────────────────────────────────

# Escolha o modelo. Opções populares:
#   "meta-llama/Llama-3-70b"
#   "mistralai/Mixtral-8x7B-v0.1"
#   "Qwen/Qwen1.5-72B"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # 8B para teste rápido

# Modo de compressão:
#   None  → sem compressão (mais rápido, mais VRAM)
#   "8bit" → moderado
#   "4bit" → menor uso de VRAM (requer bitsandbytes)
COMPRESSION = "4bit"

# Diretório onde os shards de camadas serão salvos (None = cache padrão HF)
LAYER_SHARDS_PATH = None  # Ex: "/caminho/personalizado/shards"

# Ativar prefetch (recomendado para modelos ≥ 30B — sobrepõe IO e computação)
USE_PREFETCH = False

# ─────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────

PROMPT = "Explique computação quântica em termos simples, em português."

MAX_NEW_TOKENS = 200


# ─────────────────────────────────────────────
#  FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def verificar_ambiente():
    """Verifica GPU disponível e exibe informações do ambiente."""
    print("\n📋 Verificando ambiente...")
    print(f"   Python     : {sys.version.split()[0]}")
    print(f"   PyTorch    : {torch.__version__}")

    if torch.cuda.is_available():
        nome_gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU        : {nome_gpu}")
        print(f"   VRAM total : {vram:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("   Dispositivo: Apple Silicon (MPS) — RAM unificada")
    else:
        print("   ⚠️  Nenhuma GPU detectada. Usando CPU (lento).")

    print()


def carregar_modelo(model_id: str, compression, layer_shards_path, prefetch: bool):
    """Carrega o modelo com AirLLM."""
    print(f"🚀 Carregando modelo: {model_id}")
    print(f"   Compressão : {compression or 'Nenhuma'}")
    print(f"   Prefetch   : {'Sim' if prefetch else 'Não'}")
    print("   (1ª execução baixa e fragmenta o modelo — pode demorar)\n")

    kwargs = {
        "compression": compression,
        "prefetch": prefetch,
    }
    if layer_shards_path:
        kwargs["layer_shards_saving_path"] = layer_shards_path

    model = AutoModel.from_pretrained(model_id, **kwargs)
    print("✅ Modelo carregado com sucesso!\n")
    return model


def gerar_resposta(model, prompt: str, max_new_tokens: int) -> str:
    """Executa a inferência e retorna o texto gerado."""
    print(f"💬 Prompt:\n   {prompt}\n")
    print("⏳ Gerando resposta (as camadas são carregadas sob demanda)...\n")

    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
    )

    # AutoModel retorna string ou lista — normaliza aqui
    if isinstance(output, list):
        texto = output[0]
    else:
        texto = str(output)

    return texto


def exibir_uso_vram():
    """Exibe uso atual de VRAM (apenas CUDA)."""
    if torch.cuda.is_available():
        alocado = torch.cuda.memory_allocated(0) / 1e9
        reservado = torch.cuda.memory_reserved(0) / 1e9
        print(f"\n📊 Uso de VRAM — Alocado: {alocado:.2f} GB | Reservado: {reservado:.2f} GB")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("          AirLLM — Inferência com Baixa VRAM")
    print("=" * 60)

    verificar_ambiente()

    try:
        model = carregar_modelo(
            model_id=MODEL_ID,
            compression=COMPRESSION,
            layer_shards_path=LAYER_SHARDS_PATH,
            prefetch=USE_PREFETCH,
        )

        resposta = gerar_resposta(model, PROMPT, MAX_NEW_TOKENS)

        print("─" * 60)
        print("📝 Resposta gerada:\n")
        print(resposta)
        print("─" * 60)

        exibir_uso_vram()

    except ImportError as e:
        print(f"\n❌ Dependência ausente: {e}")
        print("   Execute: pip install airllm bitsandbytes")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()
