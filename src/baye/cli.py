"""
Interactive CLI for Belief Tracking Chat

Terminal interface with rich formatting and command system.
"""

import asyncio
import sys
from typing import Optional
import json
import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

from .chat_session import ChatSession, AssistantReply, ClaimBasedReply, ClaimValidationStep, ClaimCalibrationError
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

    def __init__(self, mode: str = "legacy"):
        self.console = Console()
        self.session: Optional[ChatSession] = None
        self.mode = mode

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

        # Display mode-specific explanation
        mode_explanation = self._get_mode_explanation()
        self.console.print(Panel(
            Markdown(mode_explanation),
            title=f"📊 Mode: {self.mode.upper()}",
            border_style="yellow"
        ))

    def _get_mode_explanation(self) -> str:
        """Get mode-specific explanation banner"""
        if self.mode == "claim-based":
            return """
## Claim-Based Mode

The AI validates **specific factual claims** from its responses:
- Each claim gets a confidence estimate: **-1 to +1**
- **Negative values = disbelief** (AI thinks it's false)
- **Positive values = belief** (AI thinks it's true)
- **Zero = maximum uncertainty** (AI doesn't know)

**Scale interpretation:**
- `-0.9 to -0.5`: Strong disbelief
- `-0.5 to -0.1`: Weak disbelief
- `0.0`: Maximum uncertainty
- `0.1 to 0.5`: Weak belief
- `0.5 to 0.9`: Strong belief

**Key principle**: The AI keeps the claim text constant and uses confidence to express belief/disbelief.
            """
        else:  # legacy mode
            return """
## Legacy Mode

The AI validates beliefs using a **single belief per response**:
- Each response includes a belief with confidence: **-1 to +1**
- **Negative values = disbelief** (AI thinks it's false)
- **Positive values = belief** (AI thinks it's true)
- **Zero = maximum uncertainty** (AI doesn't know)

**Scale interpretation:**
- `-0.9 to -0.5`: Strong disbelief
- `-0.5 to -0.1`: Weak disbelief
- `0.0`: Maximum uncertainty
- `0.1 to 0.5`: Weak belief
- `0.5 to 0.9`: Strong belief

**Key principle**: The AI keeps the statement text constant and uses confidence to express belief/disbelief.

**Example**: If asked "Is Obama president?", AI will say:
- "Obama é presidente dos EUA" with confidence **-0.85** (strong disbelief)
- NOT "Obama NÃO é presidente" with confidence 0.85 (this inverts everything!)
            """

    def render_message(self, role: str, content):
        """Render a chat message"""
        if role == "user":
            self.console.print(f"[bold cyan]You:[/bold cyan] {content}")
        else:
            if isinstance(content, ClaimBasedReply):
                # Claim-based mode rendering
                self._render_claim_based_reply(content)
            elif isinstance(content, AssistantReply):
                # Legacy mode rendering - multiple steps
                for step_idx, step in enumerate(content.steps):
                    # Step header
                    if len(content.steps) > 1:
                        step_num = f"[dim]Step {step_idx + 1}/{len(content.steps)}[/dim]\n"
                    else:
                        step_num = ""

                    # Step body
                    body = step_num + step.text

                    # Step footer with metadata
                    footer = (
                        f"\n\n[dim]Belief {step.belief_id[:8]} | "
                        f"Palpite: {step.belief_value_guessed:.2f} | "
                        f"Real: {step.actual_confidence:.2f} | "
                        f"Erro: {step.error:+.2f} | "
                        f"Margem: ±{step.margin:.2f}[/dim]"
                    )

                    if abs(step.delta_requested) > 0 or abs(step.applied_delta) > 0:
                        footer += (
                            f"\n[dim]Delta pedido: {step.delta_requested:+.2f} | "
                            f"Delta aplicado: {step.applied_delta:+.2f}[/dim]"
                        )

                    panel_text = body + footer

                    self.console.print(Panel(
                        panel_text,
                        title=f"🤖 Assistant{' (Step ' + str(step_idx + 1) + ')' if len(content.steps) > 1 else ''}",
                        border_style="blue",
                        box=box.ROUNDED
                    ))

                    # Small separator between steps
                    if step_idx < len(content.steps) - 1:
                        self.console.print("")
            else:
                panel_text = str(content)
                self.console.print(Panel(
                    panel_text,
                    title="🤖 Assistant",
                    border_style="blue",
                    box=box.ROUNDED
                ))

    def _render_claim_based_reply(self, reply: ClaimBasedReply):
        """Render claim-based response with validation details"""
        # Main response text
        panel_text = reply.response_text

        # Add claim validation footer
        footer = "\n\n[dim]Claims validated:[/dim]\n"
        footer += "[dim]Legend: value? = no ground truth (system trusted LLM), ± = error without validation[/dim]\n"
        turn_score = 0.0
        for claim in reply.validated_claims:
            # Calculate points for this claim
            if claim.within_margin:
                points = abs(claim.estimate)
                turn_score += points
                points_str = f"[green]+{points:.2f}pts[/green]"

                # Color code based on error magnitude (within margin)
                error_abs = abs(claim.error)
                if error_abs < claim.margin * 0.5:
                    status = "[green]✓[/green]"  # Excellent calibration
                else:
                    status = "[yellow]✓[/yellow]"  # Good calibration (within margin)
            else:
                # Outside margin = no points
                points_str = "[red]+0.00pts[/red]"
                status = "[red]✗[/red]"  # Failed validation

            # Truncate claim if too long
            claim_text = claim.claim_content
            if len(claim_text) > 50:
                claim_text = claim_text[:47] + "..."

            # Format actual value with context
            # actual ~0.0 means K-NN has no data (uncertainty), NOT false!
            if abs(claim.actual) < 0.1:
                actual_str = f"[yellow]{claim.actual:.2f}?[/yellow]"  # ? = uncertain/no data
                err_str = f"[dim]±{abs(claim.error):.2f}[/dim]"  # Show as ± since no ground truth
            else:
                actual_str = f"{claim.actual:.2f}"
                err_str = f"{claim.error:+.2f}"

            footer += (
                f"  {status} [dim]{claim_text}[/dim] "
                f"[{claim.estimate:.2f} → {actual_str}, err: {err_str}] {points_str}\n"
            )

        # Render main response
        self.console.print(Panel(
            panel_text + footer,
            title="🤖 Assistant (claim-based)",
            border_style="blue",
            box=box.ROUNDED
        ))

        # Render statistics panel
        if self.session:
            total_score = self.session.score
            success_count = self.session.successful_claims
            failure_count = self.session.failed_claims
            total_claims = success_count + failure_count
            success_rate = (success_count / total_claims * 100) if total_claims > 0 else 0

            stats_table = Table.grid(padding=(0, 2))
            stats_table.add_column(style="cyan", justify="right")
            stats_table.add_column(style="white")

            stats_table.add_row("🎮 Turn Score:", f"[bold green]+{turn_score:.2f}pts[/bold green]")
            stats_table.add_row("📊 Total Score:", f"[bold]{total_score:.2f}pts[/bold]")
            stats_table.add_row("✓ Successes:", f"[green]{success_count}[/green]")
            stats_table.add_row("✗ Failures:", f"[red]{failure_count}[/red]")
            stats_table.add_row("📈 Success Rate:", f"{success_rate:.1f}%")

            self.console.print(Panel(
                stats_table,
                title="📊 Session Statistics",
                border_style="cyan",
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
        mode_label = "claim-based" if self.mode == "claim-based" else "legacy"
        self.console.print(f"\n[dim]Initializing belief tracker ({mode_label} mode)...[/dim]")
        self.session = ChatSession(mode=self.mode)
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

            except ClaimCalibrationError as e:
                # Claim validation error - show detailed feedback
                self.console.print(Panel(
                    str(e),
                    title="⚠️  Claim Validation Error",
                    border_style="red",
                    style="red"
                ))

            except ValueError as e:
                # These are expected errors (estimation errors, missing justifications)
                # Show them nicely to the user
                error_msg = str(e)
                if "ERRO DE ESTIMAÇÃO" in error_msg or "delta" in error_msg.lower():
                    self.console.print(Panel(
                        error_msg,
                        title="⚠️  Erro de Calibração",
                        border_style="yellow",
                        style="yellow"
                    ))
                else:
                    self.console.print(f"[red]Error: {e}[/red]")

            except Exception as e:
                self.console.print(f"[red]Unexpected Error: {e}[/red]")
                import traceback
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")


def main():
    """Entry point for CLI"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Belief Tracking Chat - Interactive AI with epistemic validation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["legacy", "claim-based"],  # Extensible for future modes
        default="legacy",
        help="Chat mode: 'legacy' (dual-agent with forced calibration) or 'claim-based' (granular claim validation)"
    )

    args = parser.parse_args()

    # Initialize CLI with selected mode
    cli = BeliefChatCLI(mode=args.mode)

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
        sys.exit(0)


if __name__ == "__main__":
    main()
