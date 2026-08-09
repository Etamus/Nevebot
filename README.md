# Nevebot

Nevebot é um bot Discord avançado escrito em Python, com IA local (LLM via llama.cpp), recursos de voz (STT com faster-whisper e TTS com Chatterbox Multilingual V3 PT-BR) e uma interface desktop para configuração em tempo real. O projeto foca em conversas naturais em português, integração com canais de voz e controle fino via UI.

## Funcionalidades

### Inteligência Artificial (LLM)
- Respostas locais usando modelos Llama (GGUF) hospedados no host.
- Personalidade única da Neve para chat e voz, com prompt editável pela interface.
- Comandos de controle via prefixo: `!casual`, `!limpar`, `!desligar`, `!bloquear` e `!desbloquear`.

### Recursos de Voz
- Transcrição (STT): usa `faster-whisper` (CTranslate2 backend) com `large-v3-turbo` em PT-BR, quantização e decodificação rápida para balancear qualidade e latência.
- Síntese de voz (TTS): usa `Chatterbox Multilingual V3` com o pacote dedicado PT-BR e clona a voz a partir de `data/voz_referencia.wav`.
- Conversão PCM alinhada a frames Opus e flush final para evitar cortes abruptos no fim do áudio.
- Reproduz TTS diretamente em canais de voz do Discord.
- Suporte a GPU (CUDA) para acelerar STT/TTS/LLM.

### Interface Web
- Servidor HTTP embutido para configuração via navegador (`web/index.html`).
- Área própria para convidar o bot a novos servidores ou removê-lo do servidor selecionado.
- Painel para editar os parâmetros expostos da LLM; opções de carregamento avisam quando exigem reinicialização.
- Ajustes em tempo real: voz, velocidade, pitch, seed, prompts, PTT e parâmetros de geração da LLM.
- A seção de comandos na UI é somente leitura (uso/descrição); edição é feita via JSON/config.
- Endpoints REST para enviar áudio, falar texto e obter/alterar config.

### Operações de Voz e UX
- Arquivo de referência de voz: `data/voz_referencia.wav` (usado para voice-clone).
- Push-To-Talk global: listener no host (Windows) via `pynput`; endpoint `/api/voz/ptt-estado` usado pela UI.
- Interface desktop nativa via `pywebview`/WebView2, com fallback para o navegador padrao.
- Gravações salvas em `gravacoes/` quando habilitado.

## Instalação

1. Clone ou copie o repositório:

```bash
git clone <url-do-repositorio>
cd Nevebot
```

2. Instale dependências (Windows):

```powershell
instalar.bat
```

- O script cria/usa um `venv`, cria as pastas locais, copia `.env.example` para `.env` se necessário, instala as dependências de `requirements.txt` e prepara o `chatterbox-tts`.
- Quando há NVIDIA disponível, o instalador prepara PyTorch com o runtime CUDA `cu128` já incluído; o NVIDIA CUDA Toolkit não é necessário. Sem NVIDIA, usa fallback CPU.
- O `llama.cpp` oficial é baixado da última release do GitHub para `llama.cpp/`.
- `faster-whisper`, Chatterbox e suas dependências são instaladas pelo instalador.
- Os pesos locais do Chatterbox PT-BR são baixados para `models/chatterbox/`.
- `install.bat` também existe como alias de compatibilidade e chama `instalar.bat`.

3. Configure variáveis de ambiente e modelos:
- Edite `.env` e preencha `DISCORD_TOKEN` e outros caminhos conforme necessário.
- Coloque o(s) modelo(s) LLM GGUF em `models/texto/` ou ajuste `LLM_MODEL_PATH`.
- Na primeira execução, `faster-whisper` baixará automaticamente o modelo STT selecionado (`large-v3-turbo` por padrão).

## Como Executar

1. Inicie o bot:

```powershell
iniciar.bat
```

2. A interface desktop abre automaticamente. Se o WebView2 não estiver disponível, o projeto abre `http://127.0.0.1:5000` no navegador padrão.

3. Use no Discord:
- Use `!casual` para manter a Neve ativa continuamente no canal selecionado.
- Em canais de voz, fale para o bot — ele transcreve e pode responder por TTS no canal.

## Arquivos de Configuração
- `data/voz_config.json`: configurações de voz (modelo STT, Chatterbox, expressividade, CFG, temperatura, velocidade, volume, seed, pitch).
- `data/config_ui.json`: configurações exibidas na UI e textos de comandos.
- `data/bloqueados.json`: lista local de usuários bloqueados; é criado automaticamente e não deve ir para o Git.

## Estrutura do Projeto

```
Nevebot/
├── nevebot.py
├── web_server.py
├── config.py
├── config_loader.py
├── instalar.bat
├── install.bat              # compatibilidade: chama instalar.bat
├── iniciar.bat
├── requirements.txt
├── cogs/
│   ├── llm_cog.py
│   └── voice_cog.py
├── services/
│   ├── stt_whisper.py        # faster-whisper wrapper (STT)
│   └── tts_chatterbox.py      # Chatterbox PT-BR TTS + voice-clone
├── data/
│   ├── config_ui.json
│   ├── voz_config.json
│   └── voz_referencia.wav
├── gravacoes/
├── logs/
├── models/
└── web/
    └── index.html
```

## Dependências Principais
- Python 3.11+
- `discord.py`
- `llama.cpp` oficial (`llama-server.exe`) para LLM GGUF
- `faster-whisper` (STT)
- `ctranslate2`, `onnxruntime`
- `chatterbox-tts` (TTS) com pesos locais PT-BR
- `pynput` (PTT global)
- `pywebview` + Microsoft Edge WebView2 Runtime (interface desktop)

Consulte `requirements.txt` para a lista completa e versões testadas.

## Notas e Recomendações
- Recomendamos GPU (CUDA) para desempenho ideal em STT/TTS/LLM.
- Default do STT é `large-v3-turbo`, usando `beam_size=1`, `temperature=0` e `int8_float16` em CUDA para melhorar compreensão em PT-BR sem pesar como o `large-v3` completo.
- O pipeline de TTS usa voice-clone diretamente em `data/voz_referencia.wav`; ao trocar o arquivo, a nova referência é detectada na próxima geração.
- Implementamos alinhamento de frames e flush no final do PCM para evitar cortes no final da fala.
