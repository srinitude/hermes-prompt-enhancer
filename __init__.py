"""
prompt-enhancer-plugin — Hermes plugin (Provider-Agnostic)
===========================================================
Registers a `pre_llm_call` hook that intercepts the last user message,
sends it to an enhancement LLM using the SAME provider the user is currently
configured for, and replaces the original message with the enhanced version.

Design principles:
- Read config at CALL TIME (inside the hook), not import time.
- Use OpenAI-compatible /chat/completions universally — Hermes providers are
  all OpenAI-compatible (including Anthropic via translation layer).
- Resolve base_url, api_key, and model from the user's current config.
- Support custom providers (venice, openrouter, any base_url).
- Graceful fallback: if enhancement fails, return {} so original passes through.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

PLUGIN_VERSION = "2.2"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config loader (call-time, not import-time)
# ---------------------------------------------------------------------------

_HERMES_CFG: Any = None
_HERMES_CFG_MTIME: float = 0.0


def _load_hermes_config() -> dict:
    """Read ~/.hermes/config.yaml at call time. Cached but invalidated on file change."""
    global _HERMES_CFG, _HERMES_CFG_MTIME
    cfg_path = Path.home() / ".hermes" / "config.yaml"
    if not cfg_path.exists():
        _HERMES_CFG = {}
        _HERMES_CFG_MTIME = 0.0
        return _HERMES_CFG

    try:
        mtime = cfg_path.stat().st_mtime
    except Exception:
        mtime = 0.0

    if _HERMES_CFG is not None and mtime == _HERMES_CFG_MTIME:
        return _HERMES_CFG  # Cache hit

    # Cache miss or file changed — re-read
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml
        _HERMES_CFG = yaml.safe_load(text) or {}
    except Exception:
        _HERMES_CFG = _minimal_yaml_parse(text)

    _HERMES_CFG_MTIME = mtime
    return _HERMES_CFG


def _minimal_yaml_parse(text: str) -> dict:
    """Parse a tiny subset of YAML sufficient for Hermes config."""
    result: dict = {}
    indent_stack: list[tuple[int, dict]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())

        while indent_stack and indent_stack[-1][0] >= indent:
            indent_stack.pop()

        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if val == "":
                new_section: dict = {}
                if indent_stack:
                    parent = indent_stack[-1][1]
                    if isinstance(parent, dict):
                        parent[key] = new_section
                else:
                    result[key] = new_section
                indent_stack.append((indent, new_section))
            elif val.startswith('"') and val.endswith('"'):
                v = val[1:-1]
                if indent_stack:
                    parent = indent_stack[-1][1]
                    if isinstance(parent, dict):
                        parent[key] = v
                else:
                    result[key] = v
            elif val.startswith("'") and val.endswith("'"):
                v = val[1:-1]
                if indent_stack:
                    parent = indent_stack[-1][1]
                    if isinstance(parent, dict):
                        parent[key] = v
                else:
                    result[key] = v
            elif val.startswith("[") and val.endswith("]"):
                items = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
                if indent_stack:
                    parent = indent_stack[-1][1]
                    if isinstance(parent, dict):
                        parent[key] = items
                else:
                    result[key] = items
            else:
                if val.lower() == "true":
                    v = True
                elif val.lower() == "false":
                    v = False
                else:
                    try:
                        v = int(val)
                    except ValueError:
                        try:
                            v = float(val)
                        except ValueError:
                            v = val
                if indent_stack:
                    parent = indent_stack[-1][1]
                    if isinstance(parent, dict):
                        parent[key] = v
                else:
                    result[key] = v
        elif stripped.startswith("- "):
            item = stripped[2:].strip().strip('"').strip("'")
            if indent_stack:
                parent = indent_stack[-1][1]
                if isinstance(parent, dict):
                    for k, v in list(parent.items()):
                        if isinstance(v, list):
                            v.append(item)
                            break

    return result


# ---------------------------------------------------------------------------
# Provider-agnostic resolution
# ---------------------------------------------------------------------------

def _resolve_enhancer_config(cfg: dict = None) -> dict:
    """
    Resolve the enhancement endpoint from the user's CURRENT Hermes config.
    Checks custom_providers first, then falls back to the model block.
    Returns a dict with: base_url, api_key, model, provider.

    Args:
        cfg: Optional config dict for testing. If None, reads from disk.
    """
    if cfg is None:
        cfg = _load_hermes_config()
    model_cfg = cfg.get("model", {})

    provider = (model_cfg.get("provider", "") or "").lower().strip()
    base_url = (model_cfg.get("base_url", "") or "").strip().rstrip("/")
    model = (model_cfg.get("default", "") or "").strip()
    api_key = (model_cfg.get("api_key", "") or "").strip()

    # ── Check custom_providers list ──────────────────────────────────────
    # If the provider name matches a custom_provider, use THAT config instead.
    # ── Check custom_providers list ──────────────────────────────────────
    # Priority: explicit provider name match > explicit base_url match > api_key heuristic.
    # We ONLY match by api_key if there is NO explicit provider name match and NO
    # explicit base_url configured — otherwise we risk overwriting a kimi-coding base_url
    # with a venice base_url just because both configs share the same api_key.
    custom_providers = cfg.get("custom_providers", [])
    matched_by_name = False
    matched_by_key = False
    if isinstance(custom_providers, list):
        for cp in custom_providers:
            cp_name = (cp.get("name", "") or "").lower().strip()
            cp_provider = (cp.get("provider", "") or cp.get("name", "") or "").lower().strip()
            # Match by provider name or by custom provider name (highest priority)
            if cp_name == provider or cp_provider == provider:
                if cp.get("base_url"):
                    base_url = cp["base_url"].strip().rstrip("/")
                if cp.get("api_key"):
                    api_key = cp["api_key"].strip()
                if cp.get("model"):
                    model = cp["model"].strip()
                _log(f"Using custom_provider '{cp_name}' for enhancement (name match)")
                matched_by_name = True
                break
        # Only fall through to api-key heuristic if no name match AND no explicit base_url
        if not matched_by_name and not base_url:
            for cp in custom_providers:
                cp_name = (cp.get("name", "") or "").lower().strip()
                cp_provider = (cp.get("provider", "") or cp.get("name", "") or "").lower().strip()
                if api_key and cp.get("api_key") == api_key:
                    if cp.get("base_url"):
                        base_url = cp["base_url"].strip().rstrip("/")
                    if cp.get("model"):
                        model = cp["model"].strip()
                    provider = cp_name or cp_provider or provider
                    _log(f"Matched custom_provider '{cp_name}' by api_key (fallback, no explicit base_url)")
                    matched_by_key = True
                    break

    # Fallback: env vars for the configured provider
    if not api_key:
        api_key = _env_key_for_provider(provider)

    # Fallback: generic env keys
    if not api_key:
        api_key = (
            os.getenv("HERMES_API_KEY", "")
            or os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("ANTHROPIC_API_KEY", "")
            or ""
        ).strip()

    # Fallback model
    if not model:
        model = os.getenv("HERMES_MODEL", "") or "anthropic/claude-sonnet-4-20250514"

    # Fallback base_url: if none configured, infer from provider name
    if not base_url:
        base_url = _infer_base_url(provider)

    # Ensure trailing slash for httpx
    if base_url:
        base_url = base_url.rstrip("/") + "/"

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "provider": provider,
    }


def _env_key_for_provider(provider: str) -> str:
    """Map provider name to its canonical env var."""
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "nous": "NOUS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "xai": "XAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "minimax-cn": "MINIMAX_CN_API_KEY",
        "opencode-go": "OPENCODE_GO_API_KEY",
        "opencode": "OPENCODE_ZEN_API_KEY",
        "kimi-coding": "KIMI_API_KEY",
        "kimi-coding-cn": "KIMI_CN_API_KEY",
        "kimi": "KIMI_API_KEY",
        "zai": "ZAI_API_KEY",
        "venice": "VENICE_API_KEY",
    }
    key_name = env_map.get(provider, "")
    if key_name:
        val = os.getenv(key_name, "")
        if val:
            return val
    # Try generic fallback for unknown providers
    provider_env = f"{provider.upper().replace('-', '_')}_API_KEY"
    return os.getenv(provider_env, "")


def _infer_base_url(provider: str) -> str:
    """Infer base_url from provider name when not explicitly configured."""
    known = {
        "anthropic": "https://api.anthropic.com/v1",
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "nous": "https://api.nousresearch.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "xai": "https://api.x.ai/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "groq": "https://api.groq.com/openai/v1",
        "mistral": "https://api.mistral.ai/v1",
        "minimax": "https://api.minimax.chat/v1",
        "minimax-cn": "https://api.minimax.chat/v1",
        "opencode-go": "https://opencode.ai/zen/go/v1",
        "opencode": "https://opencode.ai/zen/go/v1",
        "kimi-coding": "https://api.kimi.com/coding",
        "kimi-coding-cn": "https://api.kimi.com/coding",
        "kimi": "https://api.kimi.com/coding",
        "zai": "https://api.z.ai/v1",
        "venice": "https://api.venice.ai/api/v1",
    }
    return known.get(provider, "https://openrouter.ai/api/v1")


# ---------------------------------------------------------------------------
# Enhancement system prompt
# ---------------------------------------------------------------------------

_ENHANCER_SYSTEM_PROMPT = """You are an expert prompt architect. Your task is to transform rough, vague, incomplete, or under-specified user prompts into precise, structured, execution-ready prompts.

## Core Objective
Enhance the user's prompt without executing it. Your output must preserve the user's original intent while improving clarity, completeness, specificity, constraints, structure, and actionability.

## Operating Rules
1. Do not answer the prompt's task.
2. Do not produce the final artifact requested by the original prompt.
3. Only rewrite, expand, and strengthen the prompt itself.
4. Preserve all explicit requirements from the user.
5. Do not remove constraints unless they are contradictory, unsafe, or impossible.
6. Resolve ambiguity by making clearly labeled assumptions.
7. If critical information is missing, include placeholders rather than asking follow-up questions unless the prompt cannot be meaningfully enhanced without them.
8. Make the enhanced prompt usable by an AI coding agent, research agent, writing agent, design agent, or automation agent depending on context.
9. Prefer deterministic instructions over vague language.
10. Eliminate filler, redundancy, and unclear phrasing.

## Enhancement Goals
When improving a prompt, strengthen it across these dimensions:
- Objective: What exactly should be produced?
- Context: Why is it being produced, and for whom?
- Audience: Who will use or read the output?
- Scope: What is included and excluded?
- Inputs: What source material, files, URLs, APIs, tools, or references should be used?
- Output format: Markdown, YAML, JSON, code, report, checklist, specification, etc.
- Quality bar: What makes the answer good?
- Constraints: Technical, stylistic, operational, safety, brand, platform, or deployment limits.
- Verification: How should the output be checked for correctness?
- Edge cases: What uncommon states or failure modes must be handled?
- Completion criteria: What must be true before the task is considered done?

## Default Enhanced Prompt Structure
Use this structure unless the user requests another format:

```markdown
# Enhanced Prompt

## Objective
[State the exact task.]

## Context
[Explain relevant background, product, domain, or user goal.]

## Audience
[Define the intended reader, user, implementer, or evaluator.]

## Inputs
[Specify provided materials, links, files, tools, references, or placeholders.]

## Scope
### Include
- [Required item]
- [Required item]

### Exclude
- [Out-of-scope item]
- [Out-of-scope item]

## Requirements
- [Concrete requirement]
- [Concrete requirement]
- [Concrete requirement]

## Process
1. [Step one]
2. [Step two]
3. [Step three]

## Output Format
[Specify exact structure, file type, schema, sections, or formatting rules.]

## Quality Criteria
- [Measurable or inspectable quality criterion]
- [Measurable or inspectable quality criterion]

## Validation
Before finalizing, verify that:
- [Validation item]
- [Validation item]
- [Validation item]

## Assumptions
- [Explicit assumption]
- [Explicit assumption]

## Final Instruction
Return only the completed output in the requested format. Do not include explanations, commentary, or meta-notes unless explicitly requested.
```

## Handling Ambiguity
When the original prompt is ambiguous, improve it using reasonable assumptions and include them in an `Assumptions` section.

When a required decision materially changes the output, include bracketed placeholders such as:
- `[TARGET AUDIENCE]`
- `[PRODUCT NAME]`
- `[TECH STACK]`
- `[OUTPUT FORMAT]`
- `[SOURCE URLS]`
- `[BRAND STYLE]`
- `[DEPLOYMENT TARGET]`

## For Coding Prompts
When enhancing coding prompts, include:
- Tech stack and runtime constraints
- Repository assumptions
- File structure expectations
- Implementation phases
- Testing requirements
- Error handling requirements
- Security requirements
- Performance requirements
- CI/CD commands
- Acceptance criteria
- Definition of done

## For Research Prompts
When enhancing research prompts, include:
- Research question
- Hypothesis, if applicable
- Source hierarchy
- Primary vs. secondary sources
- Citation requirements
- Disconfirming evidence
- Confidence levels
- Unknowns and limitations

## For Design Prompts
When enhancing design prompts, include:
- Visual direction
- Brand attributes
- UX goals
- Product states
- Design tokens
- Component hierarchy
- Accessibility requirements
- Responsive behavior
- Interaction states
- Deliverable format

## For Business or Strategy Prompts
When enhancing business prompts, include:
- Business objective
- Target customer
- Revenue model
- Constraints
- Success metrics
- Risks
- Tradeoffs
- Recommended decision framework
- Execution plan

## Style Rules
Write enhanced prompts with:
- Clear headings
- Imperative instructions
- Specific nouns and verbs
- Minimal ambiguity
- Logical ordering
- No unnecessary prose
- No generic motivational language

## Final Output Rule
Always return only the enhanced prompt unless the user explicitly asks for explanation, critique, or alternatives.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="milliseconds")
    with open(LOG_DIR / "enhancer.log", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _build_client(base_url: str) -> httpx.Client:
    """Build an httpx client pointed at the resolved endpoint."""
    timeout = httpx.Timeout(60.0, connect=10.0)
    transport = httpx.HTTPTransport(retries=1)
    return httpx.Client(base_url=base_url, timeout=timeout, transport=transport)


def _enhance_via_api(original: str) -> Any:
    """
    Send `original` to the user's CURRENT provider and return the enhanced prompt.
    Uses OpenAI-compatible /chat/completions universally.
    """
    if not original or not original.strip():
        return None

    cfg = _resolve_enhancer_config()
    base_url = cfg["base_url"]
    api_key = cfg["api_key"]
    model = cfg["model"]
    provider = cfg["provider"]

    if not base_url:
        _log("No base_url resolved — cannot enhance")
        return None

    if not api_key:
        _log(f"No API key resolved for provider '{provider}' — cannot enhance")
        return None

    if not model:
        _log("No model resolved — cannot enhance")
        return None

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Venice-specific: use api-key header instead of Bearer
    if "venice" in provider or "venice" in base_url:
        headers["Authorization"] = f"Bearer {api_key}"
        # Venice also accepts x-api-key
        headers["x-api-key"] = api_key

    # Anthropic native: use x-api-key + anthropic-version (but Hermes uses OpenAI-compatible)
    # Since Hermes providers are OpenAI-compatible, we stick to Bearer + /chat/completions

    endpoint = "chat/completions"
    body = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": _ENHANCER_SYSTEM_PROMPT},
            {"role": "user", "content": original},
        ],
    }

    client = _build_client(base_url)

    try:
        resp = client.post(endpoint, headers=headers, json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        _log(f"Enhancement request failed: {exc}")
        return None

    try:
        if "choices" in data and len(data["choices"]) > 0:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content", "")
        else:
            _log(f"Unexpected response shape: {json.dumps(data)[:300]}")
            return None
    except (KeyError, IndexError, TypeError) as exc:
        _log(f"Failed to parse response content: {exc} — {json.dumps(data)[:300]}")
        return None

    if not content or not content.strip():
        _log("Enhancement returned empty content")
        return None

    return content.strip()


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def _ensure_first_hook(ctx, hook_name: str, hook_fn) -> None:
    """
    Register a hook and ensure it runs FIRST before any other hooks.
    Hermes calls hooks in registration order, so we re-register ourselves
    at the front if other hooks of the same type already exist.
    """
    hooks_dict = getattr(ctx, '_hooks', None)
    if hooks_dict is None:
        ctx.register_hook(hook_name, hook_fn)
        return

    existing = list(hooks_dict.get(hook_name, []))
    ctx.register_hook(hook_name, hook_fn)

    if existing:
        all_hooks = hooks_dict.get(hook_name, [])
        our_hook = all_hooks[-1] if all_hooks else None
        if our_hook is not None:
            all_hooks.remove(our_hook)
            all_hooks.insert(0, our_hook)
            hooks_dict[hook_name] = all_hooks
            _log(f"prompt-enhancer-plugin: moved pre_llm_call hook to FIRST position "
                 f"(was {len(existing)} other hook(s) registered)")


def register(ctx):
    """Hermes plugin entrypoint — wires the pre_llm_call hook FIRST."""
    _ensure_first_hook(ctx, "pre_llm_call", pre_llm_call_hook)
    _log("prompt-enhancer-plugin v2.2 registered (provider-agnostic, pre_llm_call hook active, FIRST position)")


# ---------------------------------------------------------------------------
# pre_llm_call hook
# ---------------------------------------------------------------------------

def _is_enabled() -> bool:
    """Check the opt-in env flag."""
    return os.getenv("PROMPT_ENHANCER_ENABLED", "1").lower() in ("1", "true", "yes")


def pre_llm_call_hook(
    *,
    session_id: str = "",
    user_message: str = "",
    conversation_history: list[dict] | None = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    sender_id: str = "",
    **kwargs: Any,
) -> dict | str | None:
    """
    Hermes `pre_llm_call` hook.

    Hermes invokes this with:
        session_id, user_message, conversation_history, is_first_turn,
        model, platform, sender_id

    We enhance the *current turn's user_message* via the user's CURRENT
    provider and return {"context": enhanced} for Hermes to inject into
    the user message.  Hermes appends this context to the user message
    (ephemeral, not persisted to session DB).
    """
    del kwargs  # Unused — silence linter

    if not _is_enabled():
        return {}

    # The message to enhance is the current turn's user_message
    original_content = user_message or ""
    if not original_content.strip():
        return {}

    # Skip commands and meta-prompts
    stripped = original_content.strip()
    if stripped.startswith("/"):
        return {}

    skip_prefixes = (
        "enhanced prompt",
        "skill:",
        "/skill",
        "load skill",
    )
    if any(stripped.lower().startswith(p) for p in skip_prefixes):
        return {}

    # Skip if message is already very structured (heuristic: has many markdown headers)
    header_count = original_content.count("## ") + original_content.count("### ")
    if header_count >= 3 and "objective" in original_content.lower():
        _log(f"Skipping already-structured message ({header_count} headers)")
        return {}

    _log(f"Enhancing message ({len(original_content)} chars): {original_content[:80]!r}...")

    run_id = str(uuid.uuid4())[:8]
    ts = time.time()

    enhanced = _enhance_via_api(original_content)

    elapsed = time.time() - ts

    if enhanced is None:
        _log(f"[{run_id}] Enhancement failed — using original")
        return {}

    _log(f"[{run_id}] Enhanced in {elapsed:.1f}s — {len(original_content)} → {len(enhanced)} chars")

    # Return context injection for Hermes — it appends this to the user message
    return {"context": enhanced}
