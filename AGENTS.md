# AGENTS.md

For AI agents and contributors: how 53ON AI Server works, how to call it, and how to change it.

## What it is

A self-hosted AI manager for text, image and video, in one Node.js process with no dependencies. It speaks the OpenAI and Anthropic APIs, runs local text models on two engines (llama.cpp and Strata), pictures and clips on stable-diffusion.cpp, relays cloud providers, and hands out API keys. Built and measured on AMD Ryzen AI Max (Strix Halo) with Ubuntu 24.04.

## Run it

```
node server.js            # Node.js 18+, nothing to install
```

The first start writes `config.json` from `config.example.json` and prints the admin token once. HTTPS on port 11500 when `certs/fullchain.pem` and `certs/privkey.pem` exist, plain HTTP on 11501 for the local network. The panel is `/admin`.

## Call it

- `POST /v1/chat/completions`, `/v1/completions`, `GET /v1/models` (OpenAI), `POST /v1/messages` (Anthropic). Header `Authorization: Bearer <api key>`.
- `GET /v1/models` gives each model's `context_length` (what it serves) and `max_context_length` (what it was trained for).
- Streaming: send `"return_progress": true` and a local model reports its prompt read as `prompt_progress` chunks (`total`, `cache`, `processed`, `time_ms`), from llama.cpp and from Strata alike.

### Tools the server runs itself

Declare them in `tools` as OpenAI functions; when the model calls one, the server runs it and adds the result to the reply's text. The call never reaches the client.

- `generate_image {prompt, size?, image?}` draws a picture, or edits the one in `image` (a URL from this conversation). The reply gets `![](<base>/img/<32 hex>.png)`.
- `generate_video {prompt, seconds? (4-15), image?}` films a clip with sound; `image` is its first frame. A clip takes minutes, so it is a background job:
  - the first status chunk carries its link: `delta: {status: "Making a video with <model>", video: "<base>/vid/<32 hex>.mp4"}`;
  - the clip is made and kept even if the client leaves;
  - the link answers `202 {"status":"in_progress"}` while it is made, `410` with the reason if it failed, and the MP4 (H.264 + AAC, byte ranges) once it is there; clips and pictures live 24 h;
  - if the stream is still open, the reply ends with `[🎬 video](<link>)`.

### Admin API

`/admin/api/*` with header `x-admin-token: <admin token>`, from private, loopback and VPN addresses only. The main calls: `GET state`, `GET metrics` (hardware and live slots), `GET pulse` (recent requests), `GET|POST catalog` (model cards), `POST model {model, action: "load"|"unload"}`.

## Code map

| File | What it does |
|---|---|
| `server.js` | HTTP doors, API keys and quotas, admin API, memory gate, `/img` and `/vid` |
| `cloud_relay.js` | relays every chat request (local or cloud) and runs the image and video tools |
| `chat_local.js` | local text engines: llama.cpp (`llama-server`) and Strata (`strata serve`), slots, live progress |
| `image_local.js`, `video_local.js` | stable-diffusion.cpp's `sd-server` for pictures and clips |
| `anthropic.js`, `claude_engine.js` | the Anthropic door and Claude Code as a model |
| `mem.js`, `prices.js`, `i18n.js` | free GPU memory, cloud prices, panel languages |
| `admin.html`, `lang/*.json` | the panel and its words (English and Spanish) |
| `catalog.json` | 53ON's calibration: engine arguments, sampling and generation settings per model file |

## Rules for changes

- Plain Node.js, no npm packages, one process. Code, names and comments in English; words the panel shows go in `lang/en.json` and `lang/es.json`.
- A model's settings live on its card (`config.json`, edited from the panel while the model is stopped); the server writes `config.json`, so do not edit it by hand while it runs.
- Fix causes, not symptoms; keep changes small and say why in a short comment only when it is not obvious.
- Run `node --check` on every changed `.js` before committing. Never commit `config.json`, `certs/` or keys.
