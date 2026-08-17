<img width="1400" height="350" alt="Nevebot Control Center" src="https://github.com/user-attachments/assets/a16f898b-e682-485e-9b89-8ca5cf2b09a5" />

---

Nevebot é uma plataforma de IA local integrada a um bot Discord, capaz de manter conversas por texto e voz com foco em interações curtas, naturais e responsivas em português brasileiro. A arquitetura utiliza uma LLM em formato GGUF executada via llama.cpp, reconhecimento de fala com faster-whisper e clonagem de voz por meio do Chatterbox Multilingual V3 PT-BR. Todo o processamento e a inferência de IA acontecem diretamente na máquina do usuário, garantindo maior privacidade e independência de APIs externas ou serviços pagos.

---

## Recursos

### Funcionalidades

- Executa modelos GGUF localmente pelo `llama-server.exe` oficial.
- Faz streaming das respostas e inicia o TTS assim que cada mensagem fica pronta.
- Mantém histórico curto por canal e filas independentes para evitar respostas sobrepostas.
- Permite editar os prompts de texto e voz.
- Expõe os parâmetros de modelo, contexto, GPU, batch, cache KV e sampling.
- Responde no Discord por menção, mensagem direta ou modo ativo no canal.

### Voz

- Transcreve com `faster-whisper` e `large-v3-turbo` por padrão.
- Prepara o áudio antes do STT com conversão para mono, resample para 16 kHz, remoção de offset, VAD e normalização de volume.
- Usa decodificação principal com beam 3 e uma segunda tentativa seletiva com beam 5 quando a transcrição parece instável.
- Filtra créditos de legenda e outras alucinações conhecidas do Whisper.
- Sintetiza com Chatterbox Multilingual V3 e o pacote dedicado a PT-BR.
- Clona automaticamente a voz de `data/voz_referencia.wav` e detecta a troca do arquivo na geração seguinte.
- Reproduz PCM diretamente no Discord, sem depender de FFmpeg.
- Recebe e reproduz localmente as pessoas do canal de voz, com seleção de saída e volume, sem gravar ou transcrever.
- Gera legendas SRT em tempo quase real com timestamps e identificação por pessoa a partir do canal do Discord, em um modo separado da LLM.
- Mantém os modelos descarregados na inicialização; **Iniciar modelo** carrega e pré-aquece LLM, Whisper e TTS juntos antes de liberar a conversa.

### Interface

A interface principal abre como aplicativo desktop com `pywebview` e Microsoft Edge WebView2. Quando esse renderer não está disponível, o Nevebot tenta abrir `http://127.0.0.1:5000` no navegador instalado.

As páginas atuais são:

- **Visão geral:** estado do Discord, canal de voz, LLM, reconhecimento, síntese e microfone.
- **Conversa:** chat por texto, gravação pelo microfone e push-to-talk.
- **Voz:** entrada, reconhecimento, referência e parâmetros do Chatterbox.
- **Modelo:** seleção do GGUF, parâmetros de execução, sampling e prompts.
- **Discord:** servidores, conexão, monitor local, envio de mensagens e transcrição SRT do canal.
- **Comandos:** referência dos comandos disponíveis no bot.

## Requisitos

- Windows 10 ou Windows 11 de 64 bits.
- Python 3.11 de 64 bits. Quando não estiver instalado, o `instalar.bat` tenta prepará-lo pelo `winget`.
- Uma aplicação de bot criada no Discord Developer Portal.
- O intent privilegiado **Message Content Intent** habilitado para o bot.
- Um modelo de texto no formato GGUF compatível com a versão atual do `llama.cpp`.
- Espaço em disco para o GGUF, Whisper, Chatterbox e dependências do PyTorch.
- GPU NVIDIA recomendada para baixa latência. O projeto também possui fallback para CPU, com desempenho menor.

O NVIDIA CUDA Toolkit global não é necessário. O instalador usa o runtime CUDA incluído nos pacotes do PyTorch quando encontra um driver NVIDIA compatível.

## Instalação

1. Clone o repositório:

```powershell
git clone https://github.com/Etamus/Nevebot.git
cd Nevebot
```

2. Execute o instalador:

```powershell
instalar.bat
```

O `instalar.bat`:

- cria o ambiente virtual em `venv/`;
- instala Python 3.11 pelo `winget` quando não encontra uma instalação compatível;
- cria as pastas locais usadas pelo projeto;
- copia `.env.example` para `.env` quando necessário;
- instala WebView2 via `winget` quando possível;
- instala PyTorch com CUDA `cu128` em máquinas NVIDIA ou usa o pacote para CPU;
- instala as dependências fixadas em `requirements.txt`;
- baixa a versão oficial mais recente do `llama.cpp` para `llama.cpp/`;
- baixa antecipadamente o modelo configurado do `faster-whisper` para `models/whisper/`;
- baixa os pesos do Chatterbox PT-BR para `models/chatterbox/`;
- executa um diagnóstico final das dependências, binários, pesos e arquivos obrigatórios.

O instalador prepara todos os componentes públicos do projeto. Token do Discord, modelo GGUF e gravação de referência continuam sendo fornecidos pelo usuário; quando algum deles estiver ausente, o diagnóstico final mostra exatamente o que falta.

3. Abra `.env` e substitua o token de exemplo:

```dotenv
DISCORD_TOKEN=SEU_TOKEN_REAL
```

4. Coloque seu modelo GGUF em:

```text
models/texto/seu-modelo.gguf
```

O modelo de texto não é incluído no repositório. Depois de adicionar ao menos um GGUF válido, o caminho pode ser selecionado na interface ou definido por `LLM_MODEL_PATH` no `.env`.

5. Coloque a referência de voz em:

```text
data/voz_referencia.wav
```

O arquivo precisa ter pelo menos um segundo. Para uma clonagem mais estável, use uma gravação limpa, com uma única pessoa falando em PT-BR, sem música, eco, ruído forte ou vozes sobrepostas.

6. Inicie o projeto:

```powershell
iniciar.bat
```

O Nevebot abre sem carregar os modelos de inferência. Use **Iniciar modelo** na página **Visão geral** para iniciar o `llama-server` e concluir o aquecimento da LLM, do `large-v3-turbo` e do Chatterbox; o botão só indica que está pronto quando todo o pipeline de conversa terminou. O mesmo botão permite liberar a LLM depois.

## Configuração do Discord

1. Crie uma aplicação e um bot no Discord Developer Portal.
2. Em **Bot > Privileged Gateway Intents**, habilite **Message Content Intent**.
3. Coloque o token em `DISCORD_TOKEN` no `.env`.
4. Inicie o Nevebot e use **Adicionar** na página Discord da interface.
5. Selecione um servidor e um canal de voz, depois use **Conectar**.
6. Em **Escutar canal**, selecione a saída de áudio e use **Ouvir canal** para acompanhar as pessoas pelo Nevebot.
7. Em **Transcrever canal**, use **Iniciar transcrição** para gerar em `transcricoes/` um SRT com cada participante identificado.

O convite criado pela interface solicita as permissões usadas pelo projeto: ver canais, enviar mensagens, ler histórico, adicionar reações, conectar, falar e usar atividade de voz.

## Uso

O prefixo padrão é `!`. Os nomes podem ser alterados em `data/config_ui.json`.

| Comando | Função |
| --- | --- |
| `!ligar` | Mantém a Neve ativa no canal de texto atual. |
| `!desligar` | Desativa as respostas naquele canal. |
| `!limpar` | Apaga o histórico de conversa do canal. |
| `!bloquear @membro` | Impede que um membro receba respostas; restrito ao dono configurado. |
| `!desbloquear @membro` | Remove o bloqueio de um membro; restrito ao dono configurado. |

Fora do modo ativo, a Neve responde quando é mencionada e em mensagens diretas. Na página Conversa, o microfone pode ser acionado pelo botão da interface ou mantendo o **shift direito** pressionado.

Os receptores do Discord permanecem desligados até **Ouvir canal** ou **Iniciar transcrição** serem acionados. O modo **Transcrever canal** é independente da conversa por voz; enquanto ele está ativo, o chat de voz e a reprodução local do canal ficam indisponíveis para evitar disputa pelo receptor e pelo Whisper. O SRT é atualizado durante a sessão e finalizado ao parar, desconectar, trocar de canal ou desligar o Nevebot.

## Configurações

### `.env`

Contém segredos e opções de infraestrutura: token do Discord, caminho inicial do GGUF, endereço do `llama-server`, diretórios do Chatterbox e valores padrão. Consulte `.env.example` para todas as variáveis disponíveis.

### `data/config_ui.json`

Persiste prefixo, prompts, comandos e parâmetros da LLM salvos pela interface. Valores preenchidos nessa configuração têm prioridade sobre os equivalentes da LLM no `.env`.

Parâmetros de carregamento como modelo, contexto, camadas de GPU, batch, threads e cache KV entram em vigor ao desligar e ligar novamente a LLM. Parâmetros de geração e prompts são aplicados em tempo de execução.

### `data/voz_config.json`

Persiste modelo do Whisper, referência de voz, expressividade, CFG, temperatura, velocidade, volume, seed, pitch e preferências do fluxo de voz.

### `personality_prompt.json`

Contém a base estruturada de personalidade usada na composição dos prompts da Neve.

Arquivos locais como `.env`, modelos, gravações, logs, bloqueios e áudios de referência são ignorados pelo Git.

## Estrutura

```text
Nevebot/
|-- nevebot.py                  # entrada do bot e ciclo de vida
|-- desktop_ui.py               # janela pywebview e fallback de navegador
|-- web_server.py               # servidor HTTP local e pipeline voz/Discord
|-- config.py                   # configuração de runtime
|-- config_loader.py            # persistência das configurações da UI
|-- personality_prompt.json     # personalidade estruturada
|-- instalar.bat                # instalação completa
|-- iniciar.bat                 # inicialização do projeto
|-- requirements.txt
|-- cogs/
|   |-- llm_cog.py              # llama.cpp, chat e comandos
|   `-- voice_cog.py            # conexão e reprodução de voz
|-- services/
|   |-- discord_audio_monitor.py # recepção, DAVE, mixer e saída local
|   |-- discord_transcription.py # Áudio do Discord, VAD por pessoa e SRT
|   |-- discord_voice_receive.py # ativação limpa do receptor do Discord
|   |-- stt_whisper.py           # STT PT-BR com faster-whisper
|   `-- tts_chatterbox.py        # Chatterbox V3 PT-BR e clonagem
|-- scripts/
|   |-- baixar_llama_cpp.ps1
|   |-- preparar_chatterbox_ptbr.py
|   |-- preparar_whisper.py
|   `-- validar_instalacao.py
|-- data/
|   |-- config_ui.json
|   |-- voz_config.json
|   `-- voz_referencia.wav      # arquivo local, não versionado
|-- models/
|   |-- texto/                  # modelos GGUF do usuário
|   |-- whisper/                # cache do faster-whisper
|   `-- chatterbox/             # pesos locais do TTS
|-- web/
|   |-- index.html
|   |-- app.css
|   |-- favicon.png
|   `-- logo.png
|-- gravacoes/
|-- transcricoes/               # arquivos SRT locais, não versionados
`-- logs/
```

## FAQ

### O projeto não inicia

- Execute `instalar.bat --check` para obter um diagnóstico completo sem reinstalar os componentes.
- Confirme que `.env` possui um `DISCORD_TOKEN` válido.
- Confirme que existe pelo menos um `.gguf` em `models/texto/` ou no caminho definido por `LLM_MODEL_PATH`.
- Execute novamente `instalar.bat` se `venv/` ou `llama.cpp/llama-server.exe` estiverem ausentes.
- Uma única instância pode usar a interface por vez; o Nevebot bloqueia inicializações duplicadas.

### A janela desktop não abre

- O instalador tenta preparar o Microsoft Edge WebView2 Runtime automaticamente.
- Sem WebView2, a interface deve abrir no navegador em `http://127.0.0.1:5000`.
- O servidor local só fica disponível depois que o bot se conecta ao Discord.

### A LLM falha ao carregar

- Consulte `logs/llama-server.log` para falhas de inicialização.
- Consulte `logs/llama-server-runtime.log` para mensagens do servidor em execução.
- Reduza camadas de GPU, contexto, batch ou o tipo de cache KV se o modelo ultrapassar a VRAM disponível.
- Depois de alterar parâmetros de carregamento, desligue e ligue novamente o modelo na Visão geral.

### Voz lenta ou indisponível

- Confira o console e `logs/nevebot_error.log`.
- Verifique se `data/voz_referencia.wav` existe e contém fala válida.
- Em CPU, `large-v3-turbo` e Chatterbox funcionam com latência consideravelmente maior.
- Confirme no Discord se o bot tem permissão para conectar e falar no canal selecionado.
- No pywebview, permita o acesso ao microfone quando solicitado pelo Windows/WebView2.

## Privacidade

Prompts, históricos em memória, transcrição, geração de texto, clonagem e síntese de voz são processados localmente. **Escutar canal** mantém apenas uma fila curta em memória e não grava. **Transcrever canal** envia o áudio recebido somente ao Whisper local e grava o SRT em `transcricoes/`; nenhum trecho desse modo é encaminhado à LLM. O Discord recebe as mensagens e o áudio enviados aos seus canais, conforme o uso normal da plataforma. O Nevebot não exige serviços comerciais de IA.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE.txt`.
