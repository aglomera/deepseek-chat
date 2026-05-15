MODEL_ID = "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2"
OFFLOAD_FOLDER = "./model_cache"
COMPRESSION = "4bit"       # troque para None se tiver VRAM suficiente (>16GB)
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_LENGTH = 4096          # tamanho máximo do contexto de entrada
