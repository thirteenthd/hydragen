# HydraGen — high-concurrency MoE inference on 12 GB GPUs

### Eight deep conversations. 131,000 tokens each. On the GPU already in your desktop.

![context per chat: 131k tokens](https://img.shields.io/badge/context_per_chat-131k_tokens-2f81f7) ![concurrent 100k-token chats: 8](https://img.shields.io/badge/concurrent_100k_chats-8_measured-2f81f7) ![warm resume at 100k depth: ~2 s](https://img.shields.io/badge/warm_resume_at_100k-~2_s-2f81f7) ![idle chat return: 0.1 MiB](https://img.shields.io/badge/idle_chat_return-0.1_MiB-2f81f7) [![notarized experiments: 169](https://img.shields.io/badge/notarized_experiments-169_incl._failures-6e7681)](docs/CHANGES.md)

Every number measured on one RTX 3080 Ti and quality-gated before it counted;
the naive tier moves ~2 GiB to return an idle chat. No multi-GPU. No
datacenter card. No answer-quality compromise.

**Requirements:** NVIDIA Ampere GPU with **12 GB+ VRAM** (RTX 3060 Ti / 3080
Ti class) and **64 GB+ system RAM** — conversation memory streams to pinned
host RAM — plus an NVMe, CUDA toolkit 11+/12, and Python 3.10+.

**Try it** — the installer builds the engine from source and writes a default
`hydragen-local.json` (localhost, canonical stack pre-wired):

Linux:

```bash
git clone https://github.com/thirteenthd/hydragen.git && cd hydragen
./install.sh
.venv/bin/python engine/serve/server.py --engine hydragen --config hydragen-local.json
```

Windows (PowerShell):

```powershell
git clone https://github.com/thirteenthd/hydragen.git; cd hydragen
.\install.ps1
.venv\Scripts\python.exe engine\serve\server.py --engine hydragen --config hydragen-local.json
```

OpenAI-compatible API + live concurrency dashboard → `http://127.0.0.1:8181/`
(point the config's model-blob paths at your own model set first — see
Install notes).

## The headline

One fleet of agent sessions, each holding a ~100k-token conversation, on one