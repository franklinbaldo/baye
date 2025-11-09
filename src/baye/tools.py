"""
Tool system for LLM agents
All tool returns (including errors) become Facts in the fact store
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import subprocess
import tempfile
import uuid
from datetime import datetime


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_fact_content(self) -> str:
        """Convert tool result to fact content string"""
        if self.success:
            return f"[{self.tool_name}] Success: {self.output}"
        else:
            return f"[{self.tool_name}] Error: {self.error}"


class PythonTool:
    """Execute Python code safely"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.tool_uuid = f"tool_python_{uuid.uuid4()}"

    def execute(self, code: str) -> ToolResult:
        """
        Execute Python code and return result

        Args:
            code: Python code to execute

        Returns:
            ToolResult with stdout/stderr
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute with timeout
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                return ToolResult(
                    tool_name="python",
                    success=True,
                    output=result.stdout.strip() or "(no output)",
                    metadata={'returncode': 0}
                )
            else:
                return ToolResult(
                    tool_name="python",
                    success=False,
                    output=result.stdout.strip(),
                    error=result.stderr.strip(),
                    metadata={'returncode': result.returncode}
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="python",
                success=False,
                output="",
                error=f"Execution timeout ({self.timeout}s)",
                metadata={'timeout': True}
            )

        except Exception as e:
            return ToolResult(
                tool_name="python",
                success=False,
                output="",
                error=str(e),
                metadata={'exception': type(e).__name__}
            )


class QueryFactsTool:
    """Query facts from fact store"""

    def __init__(self, fact_store):
        self.fact_store = fact_store
        self.tool_uuid = f"tool_query_facts_{uuid.uuid4()}"

    def execute(
        self,
        query: Optional[str] = None,
        fact_id: Optional[str] = None,
        limit: int = 5
    ) -> ToolResult:
        """
        Query facts by semantic search or ID

        Args:
            query: Semantic query string
            fact_id: Specific fact ID to retrieve
            limit: Maximum number of results

        Returns:
            ToolResult with found facts
        """
        try:
            if fact_id:
                # Direct lookup by ID
                fact = self.fact_store.get_fact(fact_id)
                if fact:
                    output = f"Fact #{fact.seq_id}: {fact.content}\n"
                    output += f"  Confidence: {fact.confidence:.2f}\n"
                    output += f"  Created: {fact.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    output += f"  Source: {fact.source_context_id}"

                    return ToolResult(
                        tool_name="query_facts",
                        success=True,
                        output=output,
                        metadata={'fact_id': fact_id, 'found': True}
                    )
                else:
                    return ToolResult(
                        tool_name="query_facts",
                        success=False,
                        output="",
                        error=f"Fact ID {fact_id} not found",
                        metadata={'fact_id': fact_id, 'found': False}
                    )

            elif query:
                # Semantic search
                results = self.fact_store.vector_store.search_facts(
                    query=query,
                    k=limit
                )

                if results:
                    output_lines = [f"Found {len(results)} facts:"]
                    for fact_id, content, distance, metadata in results:
                        seq_id = metadata.get('seq_id', '?')
                        confidence = metadata.get('confidence', 0.0)
                        output_lines.append(
                            f"  • Fact #{seq_id} (similarity: {1-distance:.2f}, conf: {confidence:.2f})"
                        )
                        output_lines.append(f"    {content[:100]}...")

                    return ToolResult(
                        tool_name="query_facts",
                        success=True,
                        output="\n".join(output_lines),
                        metadata={'query': query, 'count': len(results)}
                    )
                else:
                    return ToolResult(
                        tool_name="query_facts",
                        success=True,
                        output="No facts found matching query",
                        metadata={'query': query, 'count': 0}
                    )

            else:
                return ToolResult(
                    tool_name="query_facts",
                    success=False,
                    output="",
                    error="Must provide either 'query' or 'fact_id'",
                    metadata={}
                )

        except Exception as e:
            return ToolResult(
                tool_name="query_facts",
                success=False,
                output="",
                error=str(e),
                metadata={'exception': type(e).__name__}
            )


class QueryBeliefsTool:
    """Query beliefs from belief tracker"""

    def __init__(self, belief_tracker):
        self.belief_tracker = belief_tracker
        self.tool_uuid = f"tool_query_beliefs_{uuid.uuid4()}"

    def execute(
        self,
        belief_id: Optional[str] = None,
        content_query: Optional[str] = None,
        limit: int = 5
    ) -> ToolResult:
        """
        Query beliefs by ID or content

        Args:
            belief_id: Specific belief ID
            content_query: Search by content similarity
            limit: Maximum number of results

        Returns:
            ToolResult with found beliefs
        """
        try:
            if belief_id:
                # Direct lookup
                belief = self.belief_tracker.graph.beliefs.get(belief_id)
                if belief:
                    pseudo_counts = self.belief_tracker.pseudo_counts.get(belief_id, (1.0, 1.0))
                    certainty = pseudo_counts[0] + pseudo_counts[1]

                    output = f"Belief: {belief.content}\n"
                    output += f"  Confidence: {belief.confidence:.2f}\n"
                    output += f"  Certainty: {certainty:.1f} (α={pseudo_counts[0]:.1f}, β={pseudo_counts[1]:.1f})\n"
                    output += f"  Context: {belief.context}"

                    return ToolResult(
                        tool_name="query_beliefs",
                        success=True,
                        output=output,
                        metadata={'belief_id': belief_id, 'found': True}
                    )
                else:
                    return ToolResult(
                        tool_name="query_beliefs",
                        success=False,
                        output="",
                        error=f"Belief ID {belief_id} not found",
                        metadata={'belief_id': belief_id, 'found': False}
                    )

            elif content_query:
                # Search by content similarity (simple for now)
                matches = []
                for bid, belief in self.belief_tracker.graph.beliefs.items():
                    if content_query.lower() in belief.content.lower():
                        matches.append((bid, belief))

                matches = matches[:limit]

                if matches:
                    output_lines = [f"Found {len(matches)} beliefs:"]
                    for bid, belief in matches:
                        pseudo_counts = self.belief_tracker.pseudo_counts.get(bid, (1.0, 1.0))
                        certainty = pseudo_counts[0] + pseudo_counts[1]

                        output_lines.append(
                            f"  • {belief.content[:80]}... "
                            f"(conf: {belief.confidence:.2f}, cert: {certainty:.1f})"
                        )

                    return ToolResult(
                        tool_name="query_beliefs",
                        success=True,
                        output="\n".join(output_lines),
                        metadata={'query': content_query, 'count': len(matches)}
                    )
                else:
                    return ToolResult(
                        tool_name="query_beliefs",
                        success=True,
                        output="No beliefs found matching query",
                        metadata={'query': content_query, 'count': 0}
                    )

            else:
                return ToolResult(
                    tool_name="query_beliefs",
                    success=False,
                    output="",
                    error="Must provide either 'belief_id' or 'content_query'",
                    metadata={}
                )

        except Exception as e:
            return ToolResult(
                tool_name="query_beliefs",
                success=False,
                output="",
                error=str(e),
                metadata={'exception': type(e).__name__}
            )


class ToolRegistry:
    """Registry of available tools"""

    def __init__(self, fact_store, belief_tracker):
        self.fact_store = fact_store
        self.belief_tracker = belief_tracker

        # Initialize tools
        self.tools = {
            'python': PythonTool(),
            'query_facts': QueryFactsTool(fact_store),
            'query_beliefs': QueryBeliefsTool(belief_tracker),
        }

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any] = None, **kwargs) -> ToolResult:
        """Execute a tool and return result"""
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
                metadata={'available_tools': list(self.tools.keys())}
            )

        tool = self.tools[tool_name]

        # Use parameters if provided, otherwise use kwargs
        tool_kwargs = parameters if parameters is not None else kwargs

        # Validate that required parameters are present
        try:
            return tool.execute(**tool_kwargs)
        except TypeError as e:
            # Tool was called with missing/wrong parameters
            error_msg = str(e)

            # Extract expected parameter names from error
            if "missing" in error_msg and "required positional argument" in error_msg:
                # e.g., "missing 1 required positional argument: 'code'"
                missing_param = error_msg.split("'")[-2] if "'" in error_msg else "unknown"

                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output="",
                    error=f"Missing required parameter: '{missing_param}'. You called tool '{tool_name}' with parameters: {tool_kwargs}. Please provide the '{missing_param}' parameter.",
                    metadata={
                        'provided_parameters': tool_kwargs,
                        'missing_parameter': missing_param
                    }
                )
            else:
                # Generic parameter error
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output="",
                    error=f"Parameter error: {error_msg}. Parameters provided: {tool_kwargs}",
                    metadata={'provided_parameters': tool_kwargs}
                )

    def get_tool_descriptions(self) -> str:
        """Get descriptions of all available tools"""
        return """
**Available Tools**:

1. **python** - Execute Python code
   Parameters:
   - code: str (Python code to execute)

   Example:
   ```json
   {"tool": "python", "code": "print(2 + 2)"}
   ```

2. **query_facts** - Search or retrieve facts
   Parameters:
   - query: str (semantic search query) OR
   - fact_id: str (specific fact ID)
   - limit: int (max results, default 5)

   Examples:
   ```json
   {"tool": "query_facts", "query": "presidente dos EUA"}
   {"tool": "query_facts", "fact_id": "abc-123-def"}
   ```

3. **query_beliefs** - Search or retrieve beliefs
   Parameters:
   - content_query: str (search by content) OR
   - belief_id: str (specific belief ID)
   - limit: int (max results, default 5)

   Examples:
   ```json
   {"tool": "query_beliefs", "content_query": "Python"}
   {"tool": "query_beliefs", "belief_id": "xyz-789"}
   ```

**IMPORTANT**: All tool results (success AND errors) are stored as Facts with complete provenance!
"""
