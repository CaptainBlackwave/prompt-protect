# Prompt Protect

A stable, modern tool for testing and hardening system prompts against LLM attacks.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google, Azure OpenAI, Ollama, AWS Bedrock
- **Comprehensive Attacks**: Jailbreak, Prompt Injection, RAG Attacks, System Prompt Extraction
- **Async Architecture**: Built with asyncio for efficient parallel testing
- **Structured Output**: JSON export for integration with CI/CD pipelines
- **Plugin System**: Easy to extend with custom attacks
- **Type-Safe**: Built with Pydantic for robust configuration management

## Installation

```bash
pip install prompt-protect
```

Or install from source:

```bash
cd prompt-protect
pip install -e .
```

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run tests against a system prompt file
prompt-protect -s system_prompt.txt
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | AWS region (default: us-east-1) |

## Usage

### Basic Usage

```bash
prompt-protect -s path/to/system_prompt.txt
```

### With Custom Models

```bash
prompt-protect -s system_prompt.txt \
    --attack-provider anthropic \
    --attack-model claude-3-sonnet-20240229 \
    --target-provider openai \
    --target-model gpt-4-turbo
```

### List Available Providers

```bash
prompt-protect list-providers
```

### List Available Attacks

```bash
prompt-protect list-attacks
```

### Output Results to JSON

```bash
prompt-protect -s system_prompt.txt -o results.json
```

## Architecture

```
prompt_protect/
├── core/           # Core functionality
│   ├── config.py   # Configuration management
│   ├── client.py   # LLM client implementations
│   └── fuzzer.py   # Main fuzzing engine
├── attacks/        # Attack implementations
│   ├── jailbreak/  # Jailbreak attacks
│   ├── injection/  # Prompt injection attacks
│   └── rag/        # RAG-specific attacks
└── cli.py          # Command-line interface
```

## Supported Attacks

### Jailbreak
- AIM Jailbreak
- DAN (Do Anything Now)
- UCAR (Universal Cartoon Avatar Refusal)
- Translation Bypass

### Prompt Injection
- Contextual Redirection
- Complimentary Transition
- Typoglycemia

### RAG Attacks
- RAG Poisoning (Hidden Parrot)

### System Prompt Extraction
- System Prompt Stealer

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy prompt_protect

# Linting
ruff check prompt_protect
```

## License

MIT License
