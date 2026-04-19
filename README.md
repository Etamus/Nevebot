# Nevebot

Nevebot é um bot Discord avançado escrito em Python, com IA local (LLM via llama.cpp), recursos de voz (STT com faster-whisper e TTS com OmniVoice) e uma interface web para configuração em tempo real. O projeto foca em conversas naturais em português, integração com canais de voz e controle fino via UI.

## Funcionalidades

### Inteligência Artificial (LLM)
- Respostas locais usando modelos Llama (GGUF) hospedados no host.
- Modos de operação:
  - **Assistente**: respostas diretas e objetivas.
  - **Lou**: persona casual e introspectiva (não admite ser IA).
- Comandos especiais (via prefixo): `!resumir`, `!estilo`, `!limpar`, `!desligar`, `!bloquear`, entre outros.

### Recursos de Voz
- Transcrição (STT): usa `faster-whisper` (CTranslate2 backend) com VAD; modelos `small`/`medium` configuráveis para balancear latência x qualidade.
- Síntese de voz (TTS): usa `OmniVoice` em modo *voice-cloning* (gera uma referência única e reutiliza para consistência vocal).
- Conversão PCM alinhada a frames Opus e flush final para evitar cortes abruptos no fim do áudio.
- Reproduz TTS diretamente em canais de voz do Discord.
- Suporte a GPU (CUDA) para acelerar STT/TTS/LLM.

### Interface Web
- Servidor HTTP embutido para configuração via navegador (`web/index.html`).
- Ajustes em tempo real: voz, velocidade, pitch, seed, prefixo, PTT, entre outros.
- A seção de comandos na UI é somente leitura (uso/descrição); edição é feita via JSON/config.
- Endpoints REST para enviar áudio, falar texto e obter/alterar config.

### Operações de Voz e UX
- Arquivo de referência de voz: `data/voz_referencia.wav` (usado para voice-clone).
- Push-To-Talk global: listener no host (Windows) via `pynput`; endpoint `/api/voz/ptt-estado` usado pela UI.
- Gravações salvas em `gravacoes/` quando habilitado.

## Instalação

1. Clone ou copie o repositório:

```bash
git clone <url-do-repositorio>
cd Nevebot
```

2. Instale dependências (Windows):

```powershell
install.bat
```

- O script cria/ativa um `venv` e instala dependências listadas em `requirements.txt`.
- `faster-whisper` e suas dependências (`ctranslate2`, `onnxruntime`) são instaladas pelo instalador.

3. Configure variáveis de ambiente e modelos:
- Copie `.env.example` para `.env` e preencha `DISCORD_TOKEN` e outros caminhos conforme necessário.
- Coloque o(s) modelo(s) LLM GGUF em `models/`.
- Na primeira execução, `faster-whisper` baixará automaticamente o modelo STT selecionado (`small`/`medium`).

## Como Executar

1. Inicie o bot:

```powershell
iniciar.bat
```

2. Abra a interface web em `http://127.0.0.1:5000` para configurar voz, PTT, prefixos e ver logs simples.

3. Use no Discord:
- Use os comandos configurados (ex.: `!assistente`, `!casual`) para trocar modos.
- Em canais de voz, fale para o bot — ele transcreve e pode responder por TTS no canal.

## Arquivos de Configuração
- `data/voz_config.json`: configurações de voz (modelo STT, instruct do TTS, velocidade, volume, seed, pitch).
- `data/config_ui.json`: configurações exibidas na UI e textos de comandos.
- `data/bloqueados.json`: lista de usuários bloqueados.

## Estrutura do Projeto

```
Nevebot/
├── nevebot.py
├── web_server.py
├── config.py
├── config_loader.py
├── install.bat
├── iniciar.bat
├── requirements.txt
├── cogs/
│   ├── llm_cog.py
│   └── voice_cog.py
├── services/
│   ├── stt_whisper.py        # faster-whisper wrapper (STT)
│   └── tts_omnivoice.py      # OmniVoice TTS + voice-clone
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
- `llama-cpp-python` (para LLM GGUF)
- `faster-whisper` (STT)
- `ctranslate2`, `onnxruntime`
- `omnivoice` (TTS) ou SDK/environment compatível
- `pynput` (PTT global)

Consulte `requirements.txt` para a lista completa e versões testadas.

## Notas e Recomendações
- Recomendamos GPU (CUDA) para desempenho ideal em STT/TTS/LLM.
- Default do STT é configurável; recomendamos `small` para bom equilíbrio entre velocidade e qualidade em PT-BR. `medium` melhora qualidade, com latência maior.
- O pipeline de TTS usa voice-clone com uma referência persistente (`data/voz_referencia.wav`) para voz consistente entre gerações.
- Implementamos alinhamento de frames e flush no final do PCM para evitar cortes no final da fala.

## Licença
Projeto pessoal; adapte conforme sua necessidade e respeite as licenças das bibliotecas utilizadas.
