#!/usr/bin/env python3
"""
================================================================================
 DAMRU AI  --  HuggingFace Space App  (Updated v3)
================================================================================
Multi-provider, self-healing, Bhartiya AI chat interface.

PROVIDERS (priority order, auto-rotate on failure):
  1. Cerebras      -- fastest free inference
  2. OpenRouter    -- many free models
  3. GitHub Models -- GPT-4o-mini etc.
  4. SambaNova     -- fast Llama-3.3-70B
  5. Together AI   -- free Llama
  6. HF Router     -- Damru's own trained model (if available)
  7. Cloudflare    -- fallback

HF SPACE SECRETS (Settings -> Variables and secrets):
  CEREBRAS_API_KEY
  OPENROUTER_KEY
  GH_MODELS_TOKEN
  SAMBANOVA_API_KEY
  TOGETHER_API_KEY
  HF_TOKEN
  CF_ACCOUNT_ID + CF_API_TOKEN
  OWN_MODEL  (optional: Damru's own fine-tuned model ID on HF)
================================================================================
"""
import os
import time
import json
import requests
import gradio as gr

# ---- Config ------------------------------------------------------------------
APP_TITLE   = "Damru AI"
APP_DESC    = "Bhartiya AI — GPT-4 ko takkar dene wala 🥁"
MAX_TOKENS  = 1024
TIMEOUT     = 60
SYSTEM_MSG  = (
    "Tu Damru hai — ek Bhartiya AI assistant. "
    "Tu helpful, smart aur thoda desi style mein baat karta hai. "
    "Hinglish mein bhi bol sakta hai. "
    "Hamesha accurate aur helpful rehna."
)

# ---- Provider definitions ----------------------------------------------------
def _env(k, d=None):
    v = os.environ.get(k)
    return v if (v is not None and v.strip()) else d


def build_providers():
    provs = []

    # 1. Cerebras -- fastest
    if _env("CEREBRAS_API_KEY"):
        provs.append({
            "name": "Cerebras",
            "url": "https://api.cerebras.ai/v1/chat/completions",
            "model": _env("CEREBRAS_MODEL", "llama-3.3-70b"),
            "key": _env("CEREBRAS_API_KEY"),
            "headers": {},
        })

    # 2. OpenRouter
    if _env("OPENROUTER_KEY"):
        provs.append({
            "name": "OpenRouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": _env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
            "key": _env("OPENROUTER_KEY"),
            "headers": {
                "HTTP-Referer": "https://huggingface.co/spaces/Damaru-ai/damru",
                "X-Title": "Damru AI",
            },
        })

    # 3. GitHub Models (renamed from GITHUB_* to avoid env var ban)
    if _env("GH_MODELS_TOKEN"):
        provs.append({
            "name": "GitHub-Models",
            "url": "https://models.github.ai/inference/chat/completions",
            "model": _env("GH_MODEL", "openai/gpt-4o-mini"),
            "key": _env("GH_MODELS_TOKEN"),
            "headers": {},
        })

    # 4. SambaNova
    if _env("SAMBANOVA_API_KEY"):
        provs.append({
            "name": "SambaNova",
            "url": "https://api.sambanova.ai/v1/chat/completions",
            "model": _env("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct"),
            "key": _env("SAMBANOVA_API_KEY"),
            "headers": {},
        })

    # 5. Together AI
    if _env("TOGETHER_API_KEY"):
        provs.append({
            "name": "Together",
            "url": "https://api.together.xyz/v1/chat/completions",
            "model": _env("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
            "key": _env("TOGETHER_API_KEY"),
            "headers": {},
        })

    # 6. Damru's own trained model via HF Router
    own = _env("OWN_MODEL")
    hf  = _env("HF_TOKEN")
    if own and hf:
        provs.append({
            "name": "Damru-Own",
            "url": "https://router.huggingface.co/v1/chat/completions",
            "model": own,
            "key": hf,
            "headers": {},
        })

    # 7. HF Router fallback (generic open model)
    if hf and not own:
        provs.append({
            "name": "HF-Router",
            "url": "https://router.huggingface.co/v1/chat/completions",
            "model": _env("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            "key": hf,
            "headers": {},
        })

    # 8. Cloudflare Workers AI
    cf_acc = _env("CF_ACCOUNT_ID")
    cf_tok = _env("CF_API_TOKEN")
    if cf_acc and cf_tok:
        provs.append({
            "name": "Cloudflare",
            "url": f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/v1/chat/completions",
            "model": _env("CF_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
            "key": cf_tok,
            "headers": {},
        })

    return provs


# ---- Provider state (cooldowns) ----------------------------------------------
_prov_cooldown = {}  # name -> cooldown_until timestamp


def _is_healthy(prov):
    return time.time() >= _prov_cooldown.get(prov["name"], 0)


def _trip(prov_name, secs=120):
    _prov_cooldown[prov_name] = time.time() + secs
    print(f"  [cool] {prov_name} cooling {secs}s", flush=True)


# ---- Chat call ---------------------------------------------------------------
def chat_with_provider(prov, messages):
    """
    Returns (text, None) on success.
    Returns (None, error_str) on failure.
    """
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {prov['key']}"}
    headers.update(prov.get("headers", {}))

    payload = {
        "model": prov["model"],
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    try:
        r = requests.post(prov["url"], headers=headers,
                          json=payload, timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            text = (data.get("choices", [{}])[0]
                       .get("message", {}).get("content", ""))
            if text and str(text).strip():
                return str(text).strip(), None
            return None, "empty response"

        # Rate limit -> short cooldown
        if r.status_code == 429:
            _trip(prov["name"], 60)
        # Auth/not found -> long cooldown
        elif r.status_code in (401, 403, 404):
            _trip(prov["name"], 3600)
        else:
            _trip(prov["name"], 120)
        return None, f"HTTP {r.status_code}"

    except requests.exceptions.Timeout:
        _trip(prov["name"], 60)
        return None, "timeout"
    except Exception as e:
        _trip(prov["name"], 60)
        return None, str(e)[:80]


# ---- Main chat function ------------------------------------------------------
PROVIDERS = build_providers()
print(f"[Damru] {len(PROVIDERS)} providers loaded: "
      f"{', '.join(p['name'] for p in PROVIDERS)}", flush=True)

if not PROVIDERS:
    print("[WARN] No providers! Add at least CEREBRAS_API_KEY or HF_TOKEN secret.",
          flush=True)


def damru_chat(message, history):
    """
    Gradio chat function.
    history = list of [user, assistant] pairs.
    """
    if not message or not message.strip():
        return ""

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for user_msg, bot_msg in (history or []):
        if user_msg:
            messages.append({"role": "user",    "content": str(user_msg)})
        if bot_msg:
            messages.append({"role": "assistant", "content": str(bot_msg)})
    messages.append({"role": "user", "content": message.strip()})

    # Try providers in order
    errors = []
    for prov in PROVIDERS:
        if not _is_healthy(prov):
            errors.append(f"{prov['name']}: cooling")
            continue

        reply, err = chat_with_provider(prov, messages)
        if reply:
            print(f"  [ok] {prov['name']}", flush=True)
            return reply
        errors.append(f"{prov['name']}: {err}")
        print(f"  [fail] {prov['name']}: {err}", flush=True)

    # All failed
    err_summary = " | ".join(errors[:3])
    return (
        "Yaar abhi mere saare engines thoda busy hain. "
        "Ek minute baad try kar! "
        f"(Status: {err_summary})"
    )


# ---- Gradio UI ---------------------------------------------------------------
with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft(primary_hue="orange"),
    css="""
    .gr-button-primary { background: linear-gradient(135deg, #ff6b35, #f7931e) !important; }
    footer { display: none !important; }
    """
) as demo:
    gr.Markdown(f"""# {APP_TITLE} 🥁\n{APP_DESC}""")

    chatbot = gr.Chatbot(
        label="Damru",
        height=500,
        show_copy_button=True,
        avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Kuch bhi poocho Damru se...",
            scale=7,
            container=False,
        )
        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear 🗑️", scale=1)
        gr.Markdown(
            "<small>Providers: "
            + " | ".join(p["name"] for p in PROVIDERS)
            + "</small>",
            scale=4,
        )

    # Provider status display
    with gr.Accordion("Provider Status", open=False):
        def get_status():
            lines = []
            for p in PROVIDERS:
                ok = _is_healthy(p)
                cd = max(0, int(_prov_cooldown.get(p["name"], 0) - time.time()))
                status = "OK" if ok else f"Cooling ({cd}s)"
                lines.append(f"**{p['name']}**: {status} | Model: `{p['model']}`")
            if not lines:
                lines.append("No providers configured! Add secrets in Space settings.")
            return "\n\n".join(lines)

        status_md = gr.Markdown(get_status())
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(fn=get_status, outputs=status_md)

    # Event handlers
    def respond(msg, chat_history):
        if not msg or not msg.strip():
            return "", chat_history
        bot_reply = damru_chat(msg, chat_history)
        chat_history = chat_history + [[msg, bot_reply]]
        return "", chat_history

    txt.submit(respond, [txt, chatbot], [txt, chatbot])
    send_btn.click(respond, [txt, chatbot], [txt, chatbot])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, chatbot])

    gr.Markdown("""
    ---
    Made with ❤️ in Bharat | Damru AI — Khud seekhne wala AI
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
