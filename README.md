# Nevebot

Nevebot é um bot Discord avançado escrito em Python, equipado com inteligência artificial (LLM via Llama), recursos de voz (transcrição com Faster-Whisper e síntese com Kokoro ONNX) e uma interface web para configuração em tempo real. Ele permite interações conversacionais, comandos especiais e integração com canais de voz.

## Funcionalidades

### Inteligência Artificial (LLM)
- Respostas inteligentes usando modelo Llama (GGUF) hospedado localmente.
- Modos de operação:
  - **Assistente**: Responde diretamente como um assistente útil.
  - **Lou**: Conversa de forma casual e introvertida, sem admitir ser IA.
- Comandos especiais:
  - `!resumir [N]`: Resume as últimas N mensagens do canal (padrão: 30).
  - `!estilo @usuario`: Análise psicológica de um usuário.
  - `!limitar @usuario`: Bloqueia um usuário de receber respostas (apenas para o "pai").
  - `!desbloquear @usuario`: Desbloqueia um usuário.
  - `!fofoca`: Ativa modo fofoca, onde o bot vê e responde sobre mensagens recentes e auditoria do servidor.

### Recursos de Voz
- **Transcrição**: Usa Faster-Whisper para transcrever áudio em tempo real de canais de voz.
- **Síntese de Voz (TTS)**: Usa Kokoro ONNX para gerar áudio WAV natural em português brasileiro.
- Integração com Discord: Reproduz TTS em canais de voz quando ativado.
- Suporte a GPU (CUDA) para aceleração.

### Interface Web
- Servidor HTTP embarcado para configuração via navegador.
- Página web (`web/index.html`) para ajustar configurações em tempo real, como prefixo de comandos, vozes, etc.
- APIs REST para obter/salvar configurações.

### Outros
- Bloqueio de usuários para controle parental.
- Gravações de voz em `gravacoes/`.
- Configurações salvas em JSON (`data/`).

## Instalação

1. **Clone ou baixe o projeto**:
   ```
   git clone <url-do-repositorio>
   cd Nevebot
   ```

2. **Execute o script de instalação**:
   - Execute `install.bat` para instalar todas as dependências Python no ambiente virtual (venv).
   - Isso ativa o venv, instala `llama-cpp-python` via wheel específica e todas as bibliotecas do `requirements.txt`.

3. **Configure o ambiente**:
   - Copie `.env.example` para `.env` e preencha as variáveis necessárias (ex.: token do Discord, caminhos de modelos).
   - Baixe os modelos necessários:
     - Modelo Llama GGUF (ex.: `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`) para `models/`.
     - Modelos Kokoro (`kokoro-v1.0.fp16.onnx`, `voices-v1.0.bin`) para `models/kokoro/`.

## Como Executar

1. **Inicie o bot**:
   - Execute `iniciar.bat` para ativar o venv e rodar `nevebot.py`.
   - O bot se conectará ao Discord e iniciará o servidor web.

2. **Acesse a interface web**:
   - Abra um navegador e vá para `http://127.0.0.1:5000` (ou a porta configurada).
   - Configure o bot via a UI.

3. **Interaja no Discord**:
   - Use comandos como `!assistente` ou `!lou` para iniciar conversas.
   - Em canais de voz, o bot pode transcrever e responder via TTS.

## Configuração

- **Arquivo `.env`**: Contém token do Discord, caminhos de modelos, etc.
- **Arquivos JSON em `data/`**:
  - `config_ui.json`: Configurações da UI web.
  - `voz_config.json`: Configurações de voz (voz padrão, velocidade, etc.).
  - `bloqueados.json`: Lista de usuários bloqueados.
- O bot previne múltiplas instâncias usando uma porta de bloqueio (47654).

## Estrutura do Projeto

```
Nevebot/
├── nevebot.py              # Ponto de entrada principal
├── web_server.py          # Servidor HTTP para configuração
├── config.py               # Validação de config
├── config_loader.py        # Carregamento de configurações
├── install.bat             # Script de instalação
├── iniciar.bat             # Script de inicialização
├── requirements.txt        # Dependências Python
├── .env.example            # Exemplo de variáveis de ambiente
├── cogs/
│   ├── llm_cog.py          # Cog para LLM e comandos
│   └── voice_cog.py        # Cog para voz (transcrição/TTS)
├── data/
│   ├── bloqueados.json     # Usuários bloqueados
│   ├── config_ui.json      # Config UI
│   └── voz_config.json     # Config voz
├── gravacoes/              # Gravações de voz
├── models/
│   ├── Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf  # Modelo LLM
│   └── kokoro/
│       ├── kokoro-v1.0.fp16.onnx
│       └── voices-v1.0.bin
├── web/
│   └── index.html          # Interface web
└── venv/                   # Ambiente virtual (criado por install.bat)
```

## Dependências

- **Python 3.11+**
- Bibliotecas principais: `discord.py`, `llama-cpp-python`, `faster-whisper`, `kokoro-onnx`, `onnxruntime`, etc.
- Ver `requirements.txt` para a lista completa.
- Requer CUDA para aceleração GPU (opcional, mas recomendado para voz e LLM).

## Notas

- O bot usa GGML/Llama para IA local, garantindo privacidade.
- Suporte a português brasileiro em voz e texto.
- Desenvolvido para uso pessoal; ajuste conforme necessário.

## Licença

Este projeto é de código aberto. Consulte os termos das bibliotecas utilizadas.</content>
<parameter name="filePath">c:\Users\Administrador\Desktop\Nevebot\README.md