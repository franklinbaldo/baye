"""
Interactive CLI for Belief Tracking Chat

Terminal interface with rich formatting and command system.
"""

import asyncio
import sys
from typing import Optional
import json

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

from .chat_session import ChatSession, AssistantReply
from .belief_tracker import BeliefTracker


class BeliefChatCLI:
    """
    Interactive CLI for belief tracking chat

    Commands:
    - /beliefs [N] - List top N beliefs (default: 10)
    - /explain <id> - Explain belief confidence
    - /feedback <id> <success|failure> - Provide feedback on belief
    - /history [N] - Show conversation history
    - /export - Export session to JSON
    - /help - Show help
    - /quit - Exit
    """

    def __init__(self):
        self.console = Console()
        self.session: Optional[ChatSession] = None

    def render_welcome(self):
        """Display welcome message"""
        welcome = """
# 🧠 Belief Tracking Chat

An interactive AI chat with **epistemic belief tracking**.

## How it works:
- Chat naturally with the AI
- The system **automatically extracts beliefs** from conversations
- Beliefs are tracked with **confidence** and **certainty** (pseudo-counts)
- **Update-on-Use**: Beliefs update when you provide feedback
- **K-NN learning**: Similar beliefs influence each other

## Commands:
- `/beliefs` - List your beliefs
- `/explain <id>` - See how a belief's confidence was calculated
- `/feedback <id> success|failure` - Update belief based on outcome
- `/history` - View conversation
- `/export` - Export session data
- `/help` - Show this message
- `/quit` - Exit

Type a message to start!
        """
        self.console.print(Panel(Markdown(welcome), title="Welcome", border_style="green"))

    def render_message(self, role: str, content):
        """Render a chat message"""
        if role == "user":
            self.console.print(f"[bold cyan]You:[/bold cyan] {content}")
        else:
            if isinstance(content, AssistantReply):
                body = content.text
                footer = (
                    f"\n\n[dim]Belief {content.belief_id[:8]} | "
                    f"Palpite: {content.belief_value_guessed:.2f} | "
                    f"Valor registrado: {content.actual_confidence:.2f} | "
                    f"Margem: ±{content.margin:.2f}[/dim]"
                )
                if abs(content.delta_requested) > 0 or abs(content.applied_delta) > 0:
                    footer += (
                        f"\n[dim]Delta pedido: {content.delta_requested:+.2f} | "
                        f"Delta aplicado: {content.applied_delta:+.2f}[/dim]"
                    )
                panel_text = body + footer
            else:
                panel_text = str(content)

            self.console.print(Panel(
                panel_text,
                title="🤖 Assistant",
                border_style="blue",
                box=box.ROUNDED
            ))

    def render_beliefs_table(self, beliefs: list, title: str = "Beliefs"):
        """Render beliefs as a table"""
        table = Table(title=title, box=box.ROUNDED)

        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")
        table.add_column("Confidence", justify="right", style="green")
        table.add_column("Certainty", justify="right", style="yellow")
        table.add_column("Context", style="magenta")

        for b in beliefs:
            # Color code confidence
            conf = b["confidence"]
            if conf > 0.7:
                conf_str = f"[bold green]{conf:.2f}[/bold green]"
            elif conf > 0.4:
                conf_str = f"[yellow]{conf:.2f}[/yellow]"
            else:
                conf_str = f"[red]{conf:.2f}[/red]"

            # Truncate content if too long
            content = b["content"]
            if len(content) > 60:
                content = content[:57] + "..."

            table.add_row(
                b["id"][:8],
                content,
                conf_str,
                f"{b['certainty']:.1f}",
                b.get("context", "general")[:15]
            )

        self.console.print(table)

    def render_explanation(self, explanation: dict):
        """Render belief explanation"""
        belief = explanation["belief"]

        # Main belief info
        self.console.print(Panel(
            f"[bold]{belief['content']}[/bold]\n\n"
            f"Confidence: {belief['confidence']:.3f}\n"
            f"Certainty: {belief['certainty']:.1f}\n"
            f"Variance: {belief['variance']:.3f}\n"
            f"Pseudo-counts: α={belief['pseudo_counts']['a']:.2f}, "
            f"β={belief['pseudo_counts']['b']:.2f}",
            title=f"Belief {belief['id'][:8]}",
            border_style="cyan"
        ))

        # K-NN neighbors
        if explanation.get("neighbors"):
            self.console.print("\n[bold]K-NN Neighbors (semantic similarity):[/bold]")
            for n in explanation["neighbors"][:5]:
                self.console.print(
                    f"  • [{n['confidence']:.2f}] {n['content'][:70]}"
                )

        # Supporters (Justifications)
        if explanation.get("supporters"):
            self.console.print("\n[bold green]Justifications (Support):[/bold green]")
            self._render_justification_tree(explanation["supporters"], indent=1)

        # Contradictors
        if explanation.get("contradictors"):
            self.console.print("\n[bold red]Contradictors:[/bold red]")
            for c in explanation["contradictors"]:
                self.console.print(
                    f"  ✗ [{c['confidence']:.2f}] {c['content'][:70]}"
                )

    def _render_justification_tree(self, supporters: list, indent: int = 0):
        """Recursively render justification tree"""
        prefix = "  " * indent + ("└─ " if indent > 0 else "")

        for s in supporters:
            # Truncate content if too long
            content = s['content'][:60] + "..." if len(s['content']) > 60 else s['content']

            self.console.print(
                f"{prefix}[bold]✓[/bold] [{s['confidence']:.2f}] {content} "
                f"[dim]({s.get('id', 'unknown')[:8]})[/dim]"
            )

            # Recursively show this belief's supporters (if it has any)
            # This creates the justification chain visualization
            if 'supporters' in s and s['supporters']:
                self._render_justification_tree(s['supporters'], indent + 1)

    def render_update(self, update):
        """Render belief update result"""
        delta = update.new_confidence - update.old_confidence
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"

        color = "green" if delta > 0 else "red" if delta < 0 else "yellow"

        self.console.print(Panel(
            f"[bold]Belief Updated[/bold]\n\n"
            f"Confidence: {update.old_confidence:.3f} → {update.new_confidence:.3f} "
            f"([{color}]{delta_str}[/{color}])\n"
            f"Pseudo-counts: α={update.pseudo_counts[0]:.2f}, β={update.pseudo_counts[1]:.2f}\n"
            f"Training loss: {update.loss:.4f}\n"
            f"Affected beliefs: {len(update.affected_beliefs)}",
            title="Update Result",
            border_style=color
        ))

        if update.affected_beliefs:
            self.console.print(
                f"[dim]Propagated to: {', '.join(bid[:8] for bid in update.affected_beliefs[:5])}[/dim]"
            )

    async def handle_command(self, command: str) -> bool:
        """
        Handle slash commands

        Returns:
            True to continue, False to quit
        """
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/quit":
            self.console.print("[yellow]Goodbye! 👋[/yellow]")
            return False

        elif cmd == "/help":
            self.render_welcome()

        elif cmd == "/beliefs":
            n = int(parts[1]) if len(parts) > 1 else 10
            beliefs = self.session.list_beliefs(top_n=n)
            self.render_beliefs_table(beliefs)

        elif cmd == "/explain":
            if len(parts) < 2:
                self.console.print("[red]Usage: /explain <belief_id>[/red]")
                return True

            belief_id_prefix = parts[1]
            # Find full ID from prefix
            matching = [
                bid for bid in self.session.tracker.graph.beliefs.keys()
                if bid.startswith(belief_id_prefix)
            ]

            if not matching:
                self.console.print(f"[red]No belief found with ID starting with {belief_id_prefix}[/red]")
            elif len(matching) > 1:
                self.console.print(f"[yellow]Multiple matches: {', '.join(m[:8] for m in matching)}[/yellow]")
            else:
                explanation = self.session.explain_belief(matching[0])
                self.render_explanation(explanation)

        elif cmd == "/feedback":
            if len(parts) < 3:
                self.console.print("[red]Usage: /feedback <belief_id> <success|failure>[/red]")
                return True

            belief_id_prefix = parts[1]
            outcome = parts[2].lower()

            if outcome not in ["success", "failure"]:
                self.console.print("[red]Outcome must be 'success' or 'failure'[/red]")
                return True

            # Find full ID
            matching = [
                bid for bid in self.session.tracker.graph.beliefs.keys()
                if bid.startswith(belief_id_prefix)
            ]

            if not matching:
                self.console.print(f"[red]No belief found with ID starting with {belief_id_prefix}[/red]")
            elif len(matching) > 1:
                self.console.print(f"[yellow]Multiple matches: {', '.join(m[:8] for m in matching)}[/yellow]")
            else:
                update = await self.session.handle_feedback(matching[0], outcome)
                self.render_update(update)

        elif cmd == "/history":
            n = int(parts[1]) if len(parts) > 1 else 10
            history = self.session.get_history(limit=n)

            self.console.print(Panel("Conversation History", style="cyan"))
            for msg in history:
                prefix = "You" if msg.role == "user" else "Assistant"
                self.console.print(f"\n[bold]{prefix}:[/bold] {msg.content}")

        elif cmd == "/export":
            export_data = self.session.export_session()
            filename = "session_export.json"

            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2)

            self.console.print(f"[green]✓ Session exported to {filename}[/green]")

        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("[dim]Type /help for available commands[/dim]")

        return True

    async def run(self):
        """Main CLI loop"""
        self.render_welcome()

        # Initialize session
        self.console.print("\n[dim]Initializing belief tracker...[/dim]")
        self.session = ChatSession()
        self.console.print("[green]✓ Ready![/green]\n")

        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_continue = await self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Process message with loading indicator
                self.console.print("[dim]Thinking...[/dim]")

                response = await self.session.process_message(user_input)

                # Render response
                self.render_message("assistant", response)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /quit to exit[/yellow]")
                continue

            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                import traceback
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")


def main():
    """Entry point for CLI"""
    cli = BeliefChatCLI()
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
        sys.exit(0)


if __name__ == "__main__":
    main()
