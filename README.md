# 53ON AI Server

**Your own OpenAI-compatible AI server, built for AMD Ryzen AI Max. Local models, cloud providers and API keys in one control panel.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420)](#recommended-setup-amd-ryzen-ai-max) [![AMD Ryzen AI Max](https://img.shields.io/badge/AMD-Ryzen%20AI%20Max-ED1C24)](#recommended-setup-amd-ryzen-ai-max) [![Node.js 18+](https://img.shields.io/badge/Node.js-18%2B-339933)](#quick-start) [![53on.com](https://img.shields.io/badge/53on.com-store-7C5CFF)](https://53on.com)

A self-hosted server that speaks the OpenAI and Anthropic APIs over [llama.cpp](https://github.com/ggml-org/llama.cpp), with a control panel to load models, catalog them, relay cloud providers and hand out API keys with quotas. It has no dependencies: plain Node.js, one folder, one process.

![53ON AI Server: the models, the model editor, the cloud catalog and the API keys](assets/screenshots.png)

> [!TIP]
> **Give your AI a memory and a body with [53ON Harness](https://github.com/enriqxxx/53on-harness).** It remembers between sessions, sees your screen, speaks, browses and works on your PC, and it uses this server as its model. Pair them and your AI runs entirely at home.
>
> [![Get 53ON Harness](https://img.shields.io/badge/Get-53ON%20Harness-7C5CFF?style=for-the-badge)](https://github.com/enriqxxx/53on-harness/releases/latest)

## Features

- **OpenAI and Anthropic compatible.** `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/v1/messages`, with streaming, tools and vision. `/v1/models` reports each model's `context_length` (what it serves) and `max_context_length` (its trained maximum).
- **Local models with llama.cpp.** Load and stop each model from the panel. A model is never stopped while it answers, and its conversations are kept between loads.
- **Model catalog.** Copy a `.gguf` into its own folder and add it with **+**. The server reads the GGUF header (architecture and trained context) and the editor sets its type, vision projector, engine arguments, capabilities (vision, video, audio, tools, reasoning) and sampling.
- **Images.** Create and edit pictures with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp); a text model draws by calling the `generate_image` tool.
- **Cloud providers.** Relay Anthropic, OpenAI, Google, DeepSeek, Groq, Alibaba, OpenRouter and more through the same API. Provider keys stay on the server, and the server's own cloud catalog says the context of each cloud model.
- **API keys with plans.** Token quotas, monthly plans, daily caps, image permission, usage per key, export and import.
- **Live monitor.** GPU use and temperature, memory, power, and every request with its speed.
- **English and Spanish,** with help on every card, and a note in About when a newer release is out on GitHub.

## Quick start

Ubuntu 24.04 with Node.js 18 or newer:

```
git clone https://github.com/enriqxxx/53on-ai-server.git
cd 53on-ai-server
node server.js
```

The first start writes `config.json` from `config.example.json` and creates the admin token. It is printed once, on a line that reads `admin token (panel /admin): adm-…`, and it is kept in `config.json` as `admin_token`. Running as a service, read it with `journalctl -u ia-server | grep "admin token"`. Then:

1. Open `http://<server>:11500/admin` from the local network and enter the token.
2. **Settings › Engine:** choose *stable* or *nightly*. The server downloads the llama.cpp build into `~/engines`, verifies its sha256, tests it and switches to it.
3. **Models:** copy each model into its own folder, `~/models/<name>/` (the `.gguf`, and its `mmproj` beside it if it sees images), then add it with **+**.
4. **Keys:** create a key for each client. Clients use `http://<server>:11500/v1` with that key.

To run it as a service, edit `User` and `WorkingDirectory` in [deploy/ia-server.service](deploy/ia-server.service) and install it:

```
sudo cp deploy/ia-server.service /etc/systemd/system/
sudo systemctl enable --now ia-server
```

## Recommended setup: AMD Ryzen AI Max

What we run on a Ryzen AI Max+ 395 mini PC with 128 GB of unified memory:

- **Ubuntu 24.04 LTS, headless, with the RAM freed for the models.** No desktop, and the services an AI box does not use are switched off:

  ```
  sudo systemctl set-default multi-user.target
  sudo systemctl disable --now snapd snapd.socket snapd.seeded upower unattended-upgrades ModemManager avahi-daemon cups bluetooth
  sudo systemctl mask --now fwupd
  sudo touch /etc/cloud/cloud-init.disabled
  ```

  With unattended upgrades off, security updates are applied by hand (`sudo apt update && sudo apt upgrade`), and any snap you still use stops with `snapd`.
- **96 GB for the GPU.** In the BIOS, set the UMA frame buffer to 96 GB, which leaves 32 GB to the system.
- **Kernel parameters.** Add them to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub` and reboot:

  ```
  amd_iommu=off amdgpu.lockup_timeout=10000,10000,10000,10000 ttm.pages_limit=27648000 ttm.page_pool_size=2097152
  ```

  `ttm.pages_limit` lets the GPU map about 105 GB of system memory, and the longer lockup timeout keeps a long prompt from being taken for a hang.
- **A recent Mesa for Vulkan (RADV).** `sudo add-apt-repository ppa:kisak/kisak-mesa && sudo apt upgrade`.
- **The llama.cpp Vulkan build** (`ubuntu-vulkan-x64`), which the panel installs from **Settings › Engine**.
- **The service file in `deploy/`** keeps the server up if a model runs out of memory (`OOMPolicy=continue`) and stops the engines with it (`KillMode=mixed`).

## Our llama.cpp

The server runs any recent official llama.cpp build. We also keep our own build, which adds:

- **Dynamic VRAM.** The context memory grows with use, like a dynamic virtual disk, instead of being reserved whole at load: on Vulkan the KV cache and the compute buffers take memory in blocks as tokens arrive (sparse buffers, mapped directly on APUs), and on Linux the CPU path does the same. Measured on a Ryzen AI Max+ 395: a 1B model with a 1M-token window starts with 0.86 GiB instead of 8.24 GiB, and a 4B model at 262K with 4.41 GiB instead of 8.94 GiB.
- **Fair scheduling.** `--prefill-budget-busy` and `--spec-max-busy-slots`: a long prompt no longer starves the users who are already getting an answer.
- **Faster MoE on Vulkan without coopmat2** ([#27332](https://github.com/ggml-org/llama.cpp/pull/27332)): reads of 32-64 tokens 9-15 % faster on the Ryzen AI Max.
- **Slot save and restore with KV tensors over 1 GiB on Vulkan** ([#25584](https://github.com/ggml-org/llama.cpp/pull/25584)), needed for million-token windows.
- **Multimodal MTP for qwen4exp models** ([#30316](https://github.com/ggml-org/llama.cpp/pull/30316)): the draft model accepts image embeddings.

Each change is dropped as soon as upstream llama.cpp includes it.

## Configuration

Everything is set from the panel and saved to `config.json`; while the server runs, change it from the panel, not by hand. The panel and its API answer only on the local network: a request that comes from the internet through the router gets `404`. For HTTPS, put `fullchain.pem` and `privkey.pem` in `certs/`: the server then serves HTTPS on port 11500 and plain HTTP on 11501.

## Works with 53ON Harness

[53ON Harness](https://github.com/enriqxxx/53on-harness) gives your AI a memory that stays and hands, eyes and voice on your own PC, and it connects to this server as its model. The rest of 53ON, from WhatsApp answered by your AI to your own store, is at [53on.com](https://53on.com).

## Acknowledgments

53ON AI Server stands on the work of others, and we thank them:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml-org and its community, the engine that runs every local model here.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) by leejet, the engine behind image creation and editing.
- [LiteLLM](https://github.com/BerriAI/litellm), whose public table of model prices tells the server what each cloud request costs.
- [Strata](https://github.com/Niko1221/Strata), whose model library and file discovery inspired the catalog editor.
- The AMD Linux community: the Mesa RADV developers and the [kisak-mesa](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa) PPA, which keep Vulkan fast on Ryzen AI Max.

## Security

If you find a vulnerability, please report it privately through this repository's Security tab ("Report a vulnerability") instead of opening a public issue.

## License

The code is [MIT](LICENSE). The 53ON name and logo are trademarks of VCORPS LLC and are not covered by that license: see [NOTICE](NOTICE).
