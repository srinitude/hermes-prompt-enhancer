# prompt-enhancer-plugin

A Hermes Agent plugin (v2.2) that enhances user prompts via a `pre_llm_call` hook before they reach the LLM.

## Installation

### Option 1: Clone into plugins directory
```bash
cd ~/.hermes/plugins
git clone https://github.com/YOUR_USERNAME/prompt-enhancer-plugin.git
# Or if you already have the repo elsewhere:
ln -s /path/to/prompt-enhancer-plugin ~/.hermes/plugins/prompt-enhancer-plugin
```

### Option 2: Copy files manually
```bash
mkdir -p ~/.hermes/plugins/prompt-enhancer-plugin
cp __init__.py plugin.yaml README.md ~/.hermes/plugins/prompt-enhancer-plugin/
```

### Option 3: Use the install script
```bash
bash install.sh
```

### Enable in Hermes config

Edit `~/.hermes/config.yaml`:
```yaml
plugins:
  enabled:
    - prompt-enhancer-plugin   # Must be first if other plugins also use pre_llm_call
```

Or use the CLI:
```bash
hermes plugins enable prompt-enhancer-plugin
```

### Restart required

Plugins are cached at startup. Start a **new session**:
```bash
hermes /reset
# or exit and restart hermes
```

## Configuration

Set via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PROMPT_ENHANCER_ENABLED` | `1` | Set to `0` to disable |
| `PROMPT_ENHANCER_MODEL` | `anthropic/claude-sonnet-4-20250514` | Enhancement LLM |
| `PROMPT_ENHANCER_PROVIDER` | auto | `anthropic`, `openai`, `openrouter` |
| `PROMPT_ENHANCER_API_KEY` | _(falls back to provider env var)_ | API key |
| `PROMPT_ENHANCER_BASE_URL` | _(provider default)_ | Custom endpoint |

## Verification

After starting a new session:
```bash
cat ~/.hermes/plugins/prompt-enhancer-plugin/logs/enhancer.log
```

You should see:
```
prompt-enhancer-plugin v2.2 registered (provider-agnostic, pre_llm_call hook active, FIRST position)
Enhancing message (42 chars): 'your prompt here'...
[...] Enhanced in Xs — 42 → 3245 chars
```

## Development

### Running tests
```bash
cd ~/.hermes/plugins/prompt-enhancer-plugin
python3 test_plugin.py
```

Or with mise:
```bash
mise run test
```

### CI
Tests run automatically via GitHub Actions on push/PR.

## Troubleshooting

See [references/debugging-401-and-hook-returns.md](references/debugging-401-and-hook-returns.md) for common issues.
