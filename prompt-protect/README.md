# Prompt Protect

A stable, modern tool for testing and hardening system prompts against LLM attacks.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google, Azure OpenAI, Ollama, AWS Bedrock
- **Comprehensive Attacks**: Jailbreak, Prompt Injection, RAG Attacks, System Prompt Extraction
- **Async Architecture**: Built with asyncio for efficient parallel testing
- **Structured Output**: JSON export for integration with CI/CD pipelines
- **Plugin System**: Easy to extend with custom attacks
- **Type-Safe**: Built with Pydantic for robust configuration management
- **Evolutionary Fuzzing**: Semantic mutation with feedback loops
- **Advanced Scoring**: 0-10 jailbreak scoring with LLM-as-judge
- **Multi-Turn Attacks**: Stateful chained attacks with context tracking
- **Smart Caching**: SQLite-based prompt-response caching

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

## Advanced Features

### Evolutionary Fuzzing (Semantic Mutation)

Instead of static templates, Prompt Protect uses an "Attacker LLM" to mutate attack prompts based on feedback. If an attack fails, the engine automatically tries:
- Semantic variations (rephrasing)
- Language shifts (lower-resource languages)
- Encoding transforms (Base64, ROT13, hex)
- Role-play variations
- Chain-of-thought prefixes

```python
from prompt_protect import MutationEngine, EvolutionaryFuzzer

fuzzer = EvolutionaryFuzzer(
    target_client=target_llm,
    mutation_engine=mutation_engine,
    evaluator=evaluator,
    max_iterations=10,
    target_score=8.0,
)

result = await fuzz(fuzz, initial_prompt, system_prompt)
```

### Jailbreak Scoring (0-10)

Uses a multi-layered evaluation approach:
1. **Keyword Detection**: Fast refusal keyword matching
2. **LLM-as-Judge**: Uses a small model to grade responses
3. **Semantic Similarity**: Measures deviation from system prompt

```python
from prompt_protect import Evaluator, RefusalLevel

evaluator = Evaluator(evaluator_client=llm)
result = await evaluator.evaluate(attack_prompt, response, system_prompt)

print(f"Score: {result.score}/10")  # 0=hard refusal, 10=full bypass
print(f"Level: {result.refusal_level}")  # HARD_REFUSAL, PARTIAL_REFUSAL, NO_REFUSAL
```

### Multi-Turn (Chained) Attacks

Supports complex attack strategies:
- **Trust Building**: Build context over several turns before pivoting
- **Fragmented Injection**: Split payload across multiple inputs
- **RAG Shadowing**: Poison vector database first

```python
from prompt_protect import StateManager, AttackStrategy

state = StateManager()
chain = state.create_trust_building_chain(
    chain_id="attack1",
    system_prompt=system_prompt,
    benign_topics=["Tell me about Python", "How does ML work?"],
)

# Execute turns and check if pivot is warranted
if state.should_pivot_to_attack(chain, min_trust_turns=3):
    attack_prompt = state.generate_attack_prompt(chain, final_payload)
```

### Smart Caching

SQLite-based caching to avoid redundant API calls:

```python
from prompt_protect import Cache, CachedClient

cache = Cache(db_path=".prompt_protect_cache.db", ttl_hours=168)
cached_client = CachedClient(llm_client, cache)
```

### Early Exit (Short-Circuiting)

Automatically stops attack branches when safety filters trigger:

```python
result = await fuzzer.run_attack(
    attack_name="aim_jailbreak",
    attack_type=AttackType.AIM_JAILBREAK,
    attack_prompts=prompts,
    early_exit_callback=lambda: should_stop,
)
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
├── core/
│   ├── config.py       # Configuration management (Pydantic)
│   ├── client.py       # LLM client implementations (async)
│   ├── fuzzer.py      # Main fuzzing engine (parallel execution)
│   ├── evaluator.py   # Response grading (0-10 scoring)
│   ├── state.py       # Multi-turn conversation state
│   ├── mutation.py    # Evolutionary fuzzing engine
│   └── cache.py       # SQLite caching layer
├── attacks/
│   ├── base.py        # Attack plugin system
│   └── jailbreak/     # Attack implementations
└── cli.py             # Typer CLI
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

## API Usage

### Basic Fuzzing

```python
from prompt_protect import Fuzzer, FuzzerConfig, ProviderConfig, Provider, AppSettings

# Configure
attack_cfg = ProviderConfig(provider=Provider.OPENAI, model="gpt-4o-mini")
target_cfg = ProviderConfig(provider=Provider.OPENAI, model="gpt-4o-mini")

config = FuzzerConfig(
    attack_provider=attack_cfg,
    target_provider=target_cfg,
    num_attempts=3,
    num_threads=4,
    system_prompt="Your system prompt here",
)

settings = AppSettings()  # Loads from environment

# Run
fuzzer = Fuzzer(config, settings)
report = await fuzzer.run_fuzzer(attack_registry)

print(f"Breaches: {report.total_breaches}")
print(f"Resilient: {report.total_resilient}")
```

### With Advanced Features

```python
from prompt_protect import (
    Fuzzer, Cache, Evaluator, MutationEngine,
    StateManager, EvolutionaryFuzzer
)

# Initialize components
cache = Cache()
evaluator = Evaluator(evaluator_client=llm)
mutation_engine = MutationEngine(mutation_llm_client=llm)
state = StateManager()

# Create fuzzer with all features
fuzzer = Fuzzer(
    config, 
    settings, 
    cache=cache,
    evaluator=evaluator,
)

# Or use evolutionary fuzzing
evo_fuzzer = EvolutionaryFuzzer(
    target_client=target_llm,
    mutation_engine=mutation_engine,
    evaluator=evaluator,
)
```

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
