"""CLI for Prompt Protect."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .core.config import Provider, FuzzerConfig, AppSettings
from .core.client import ProviderConfig
from .core.fuzzer import Fuzzer
from .attacks import get_attack_registry

# Import attacks to register them
from .attacks.jailbreak import AIMJailbreakAttack

app = typer.Typer(
    name="prompt-protect",
    help="A stable tool for testing and hardening system prompts against LLM attacks",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    system_prompt: Path = typer.Option(
        ...,
        "--system-prompt",
        "-s",
        help="Path to file containing the system prompt to test",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    attack_provider: Provider = typer.Option(
        Provider.OPENAI,
        "--attack-provider",
        "-a",
        help="Provider for the attack LLM",
    ),
    attack_model: str = typer.Option(
        "gpt-4o-mini",
        "--attack-model",
        help="Model for the attack LLM",
    ),
    target_provider: Provider = typer.Option(
        Provider.OPENAI,
        "--target-provider",
        "-t",
        help="Provider for the target LLM",
    ),
    target_model: str = typer.Option(
        "gpt-4o-mini",
        "--target-model",
        help="Model for the target LLM",
    ),
    num_attempts: int = typer.Option(
        3,
        "--num-attempts",
        "-n",
        help="Number of attack attempts per test",
    ),
    num_threads: int = typer.Option(
        4,
        "--num-threads",
        help="Number of parallel threads",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON format)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """Run prompt security tests against a system prompt."""
    
    # Load settings from environment
    settings = AppSettings()
    
    # Load system prompt from file
    try:
        prompt_text = system_prompt.read_text()
    except Exception as e:
        console.print(f"[red]Error reading system prompt file: {e}[/red]")
        raise typer.Exit(1)
    
    # Build configuration
    attack_provider_config = ProviderConfig(
        provider=attack_provider,
        model=attack_model,
    )
    target_provider_config = ProviderConfig(
        provider=target_provider,
        model=target_model,
    )
    
    fuzzer_config = FuzzerConfig(
        attack_provider=attack_provider_config,
        target_provider=target_provider_config,
        num_attempts=num_attempts,
        num_threads=num_threads,
        system_prompt=prompt_text,
    )
    
    # Show banner
    console.print(Panel.fit(
        "[bold cyan]Prompt Protect[/bold cyan] - System Prompt Security Tester",
        border_style="cyan",
    ))
    
    # Check API keys
    if not _check_api_keys(settings, attack_provider, target_provider):
        console.print("[red]Error: Missing required API keys. Please set the appropriate environment variables.[/red]")
        raise typer.Exit(1)
    
    # Run fuzzer
    try:
        fuzzer = Fuzzer(fuzzer_config, settings)
        attack_registry = get_attack_registry()
        
        console.print("[yellow]Running security tests...[/yellow]")
        report = asyncio.run(fuzzer.run_fuzzer(dict(attack_registry)))
        
        # Display results
        _display_results(report, console)
        
        # Save output if requested
        if output:
            _save_results(report, output)
            console.print(f"[green]Results saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error running fuzzer: {e}[/red]")
        if verbose:
            raise
        raise typer.Exit(1)


@app.command()
def list_providers():
    """List all supported LLM providers."""
    table = Table(title="Supported Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Environment Variable")
    table.add_column("Notes")
    
    providers = [
        ("openai", "OPENAI_API_KEY", "OpenAI models (GPT-4, GPT-3.5)"),
        ("anthropic", "ANTHROPIC_API_KEY", "Anthropic Claude models"),
        ("google", "GOOGLE_API_KEY", "Google Gemini models"),
        ("azure_openai", "AZURE_OPENAI_API_KEY", "Microsoft Azure OpenAI"),
        ("ollama", "None", "Local Ollama models"),
        ("aws_bedrock", "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY", "AWS Bedrock models"),
    ]
    
    for name, env_var, notes in providers:
        table.add_row(name, env_var, notes)
    
    console.print(table)


@app.command()
def list_attacks():
    """List all available attack tests."""
    registry = get_attack_registry()
    
    table = Table(title="Available Attacks")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Severity", style="yellow")
    table.add_column("Description")
    
    for name, attack_class in registry:
        # Create instance to get metadata (would need init params in real impl)
        table.add_row(
            name,
            "jailbreak",  # Would come from metadata
            "high",       # Would come from metadata
            "Attack description",  # Would come from metadata
        )
    
    console.print(table)


def _check_api_keys(settings: AppSettings, attack_provider: Provider, target_provider: Provider) -> bool:
    """Check if required API keys are set."""
    required = set([attack_provider, target_provider])
    
    # Remove providers that don't need API keys
    required.discard(Provider.OLLAMA)  # Local, no key needed
    
    for provider in required:
        if provider == Provider.OPENAI and not settings.openai_api_key:
            console.print("[yellow]Warning: OPENAI_API_KEY not set[/yellow]")
            return False
        elif provider == Provider.ANTHROPIC and not settings.anthropic_api_key:
            console.print("[yellow]Warning: ANTHROPIC_API_KEY not set[/yellow]")
            return False
        elif provider == Provider.GOOGLE and not settings.google_api_key:
            console.print("[yellow]Warning: GOOGLE_API_KEY not set[/yellow]")
            return False
        elif provider == Provider.AZURE_OPENAI and not settings.azure_openai_api_key:
            console.print("[yellow]Warning: AZURE_OPENAI_API_KEY not set[/yellow]")
            return False
        elif provider == Provider.AWS_BEDROCK:
            if not settings.aws_access_key_id or not settings.aws_secret_access_key:
                console.print("[yellow]Warning: AWS credentials not set[/yellow]")
                return False
    
    return True


def _display_results(report, console: Console):
    """Display fuzzer results."""
    # Summary
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Target Model: {report.target_model}")
    console.print(f"  Attack Model: {report.attack_model}")
    console.print(f"  Total Attacks: {report.total_attacks}")
    console.print(f"  Breaches: [red]{report.total_breaches}[/red]")
    console.print(f"  Resilient: [green]{report.total_resilient}[/green]")
    console.print(f"  Errors: [yellow]{report.total_errors}[/yellow]")
    console.print(f"  Duration: {report.duration_seconds:.2f}s")


def _save_results(report, output: Path):
    """Save results to JSON file."""
    import json
    
    # Convert report to dict
    results = {
        "timestamp": report.timestamp,
        "target_model": report.target_model,
        "attack_model": report.attack_model,
        "total_attacks": report.total_attacks,
        "total_breaches": report.total_breaches,
        "total_resilient": report.total_resilient,
        "total_errors": report.total_errors,
        "duration_seconds": report.duration_seconds,
        "attacks": [],
    }
    
    for attack in report.attack_results:
        results["attacks"].append({
            "name": attack.attack_name,
            "type": attack.attack_type.value,
            "breach_count": attack.breach_count,
            "resilient_count": attack.resilient_count,
            "error_count": attack.error_count,
            "duration_seconds": attack.duration_seconds,
        })
    
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
