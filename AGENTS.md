# Agentic Ticker Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-09-21

## Active Technologies
- Bash scripting (main)
- Markdown templates
- Git workflow support
- Python 3.11 + pandas, numpy, yfinance, feedparser, pydantic, Streamlit, Google Gemini API
- N/A (in-memory processing with no persistent storage)
- Python 3.11 + Streamlit, Google Gemini API, yfinance, pycoingecko, pandas, numpy, plotly, requests, ddgs (001-code-refactoring-modularization)
- In-memory processing with no persistent storage (001-code-refactoring-modularization)

## Project Structure
```
src/
tests/

.specify/
  memory/
  scripts/bash/
  templates/

.opencode/
  command/

specs/[###-feature]/
```

## Commands
```bash
# Use virtual environment for all commands
source .venv/bin/activate

# Run all tests and linting
.specify/scripts/bash/check-prerequisites.sh

# Run single test (when implemented)
pytest tests/unit/test_specific.py -v

# Install packages (use venv)
pip install -r requirements.txt
```

## Code Style
Bash: Follow shellcheck conventions, use set -euo pipefail, quote variables

## Recent Changes
- 001-code-refactoring-modularization: Added Python 3.11 + Streamlit, Google Gemini API, yfinance, pycoingecko, pandas, numpy, plotly, requests, ddgs
- Added Python 3.11 + pandas, numpy, yfinance, feedparser, pydantic, Streamlit, Ollama
- main: Added spec-driven development workflow

## ⚠️ CRITICAL: Configuration Policy
**THIS REPOSITORY USES ENVIRONMENT VARIABLES ONLY**
- **REQUIRED**: Use environment variables for all configuration (especially for Streamlit Cloud)
- **OPTIONAL**: Use .env file for local development (copy from .env.template)
- All API keys must be set as environment variables

## Tool Usage Requirements
- **Always use appropriate tools** to verify changes before considering work complete
- **Before building**, use `Read` tool to verify file contents and `Glob`/`Grep` to find related files
- **After building**, use `List` tool to verify build outputs and `Bash` to test commands
- **For complex debugging**, use `sequential-thinking_sequentialthinking` to analyze problems systematically
- **For architectural changes**, use knowledge graph tools (`memory_create_entities`, `memory_create_relations`) to document changes
- **When researching solutions**, use `searxng-bridge_search` or `context7_get_library_docs` for external information
- **Always verify** that changes work end-to-end by testing the actual functionality
- **Use subagents proactively** to divide complex tasks into parallel flows for concurrent execution
- **Use MCP tools proactively** rather than reactively to maintain context and improve efficiency
- **Execute independent tasks in parallel using subagents** whenever possible to reduce overall completion time
- **ALWAYS use virtual environment**: Run `source .venv/bin/activate` before any Python commands or package installations

## Available MCP Tools

### Knowledge Graph & Memory Management Tools
These tools help maintain a persistent knowledge graph about the codebase, architecture, and development context.

- **`memory_create_entities`**: Create new entities in the knowledge graph. Use when introducing new concepts, components, or architectural elements that need to be tracked.
- **`memory_create_relations`**: Create relationships between entities. Use to document connections between components, dependencies, or architectural relationships.
- **`memory_add_observations`**: Add new observations to existing entities. Use to update or add new information about existing components, features, or code patterns.
- **`memory_delete_entities`**: Remove entities from the knowledge graph. Use when removing deprecated features or components that are no longer relevant.
- **`memory_delete_observations`**: Remove specific observations from entities. Use to correct outdated information or remove irrelevant details.
- **`memory_delete_relations`**: Remove relationships between entities. Use when dependencies are removed or architectural relationships change.
- **`memory_read_graph`**: Read the entire knowledge graph. Use to get a complete overview of the documented system architecture and relationships.
- **`memory_search_nodes`**: Search for nodes in the knowledge graph. Use to find specific components, features, or architectural patterns by keyword.
- **`memory_open_nodes`**: Open specific nodes by name. Use to retrieve detailed information about known entities.

### Problem-Solving & Analysis Tools
These tools provide structured thinking and problem-solving capabilities for complex development tasks.

- **`sequential-thinking_sequentialthinking`**: A detailed tool for dynamic and reflective problem-solving through thoughts. This tool helps analyze problems through a flexible thinking process that can adapt and evolve. Each thought can build on, question, or revise previous insights as understanding deepens. Use for breaking down complex problems, planning and design with room for revision, analysis that might need course correction, problems where the full scope might not be clear initially, and tasks that need to maintain context over multiple steps.

### External API & Documentation Tools
These tools provide access to external APIs and documentation for enhanced development capabilities.

- **`context7_resolve_library_id`**: Resolve package/product names to Context7-compatible library IDs. Use before accessing documentation to get the correct library identifier.
- **`context7_get_library_docs`**: Fetch up-to-date documentation for libraries. Use to get current API documentation, examples, and usage patterns for external dependencies.
- **`searxng-bridge_search`**: Perform web searches using SearxNG. Use for researching solutions, finding documentation, or investigating best practices.

### Subagent Tools
These tools enable parallel execution of tasks by launching specialized subagents to handle specific aspects of complex work.

- **`task`**: Launch specialized subagents to handle complex, multi-step tasks autonomously. Use to divide large tasks into parallel flows that can execute concurrently for improved efficiency. Available agent types include:
  - **general**: General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you.
  
  When using the Task tool, always specify a subagent_type parameter to select which agent type to use. Launch multiple agents concurrently whenever possible to maximize performance.

### Tool Usage Guidelines
- **Knowledge Graph Tools**: Use proactively when working on new features, refactoring, or architectural changes to maintain an accurate representation of the system.
- **Problem-Solving Tools**: Use for complex multi-step tasks, architectural decisions, debugging complex issues, or when needing structured thinking processes.
- **Documentation Tools**: Use when working with external libraries, APIs, or when needing current documentation for dependencies.
- **Search Tools**: Use when researching solutions, investigating issues, or needing external information.
- **Subagents**: Use proactively to divide complex tasks into parallel flows, executing independent subtasks concurrently for improved efficiency.
- **Integration**: These tools should be used as part of the normal development workflow to maintain context and access current information.

### When to Use Each Tool
- **Starting new feature**: Use `memory_create_entities` to document new components, then `memory_create_relations` to show how they connect to existing architecture.
- **Complex problem-solving**: Use `sequential-thinking_sequentialthinking` for breaking down complex tasks, architectural decisions, or multi-step debugging processes.
- **Refactoring**: Use `memory_search_nodes` to find related components, update observations with `memory_add_observations`, and adjust relations as needed.
- **Debugging**: Use `memory_search_nodes` to find related components and `searxng-bridge_search` to research solutions. For complex debugging, use `sequential-thinking_sequentialthinking` to systematically analyze the problem.
- **Learning codebase**: Use `memory_read_graph` to understand overall architecture and `memory_search_nodes` to find specific areas.
- **External dependencies**: Use `context7_resolve_library_id` followed by `context7_get_library_docs` to get current documentation.
- **Architecture planning**: Use `sequential-thinking_sequentialthinking` to think through complex design decisions and their implications.
- **Session resumption**: Use `memory_read_graph` to understand previous work context, then `memory_create_entities` and `memory_create_relations` to document new progress and maintain continuity across sessions.

### Proactive Tool Usage and Parallel Execution
To maximize efficiency and maintain high-quality development practices, agents should proactively use MCP tools and subagents whenever possible:

- **Proactive Knowledge Graph Updates**: Use knowledge graph tools (`memory_create_entities`, `memory_create_relations`, `memory_add_observations`) immediately when starting new work to document components, relationships, and observations as you go, rather than waiting until the end.

- **Parallel Task Execution**: When working on complex features that involve multiple independent tasks, use the `task` tool to launch subagents for each independent task. This allows for concurrent execution and faster completion times.

- **Early Research and Documentation**: Use `context7_resolve_library_id` and `context7_get_library_docs` proactively when working with external dependencies, even before you need the specific information, to have documentation ready when needed.

- **Continuous Problem Analysis**: For complex problems, initiate `sequential-thinking_sequentialthinking` early in the process and update it as you work, rather than waiting until you're stuck to begin structured thinking.

- **Concurrent File Operations**: When you need to read, search, or modify multiple files that don't depend on each other, use tools like `Read`, `Glob`, and `Grep` in parallel batches rather than sequentially.

- **Parallel Testing and Verification**: When possible, run tests, linting, and other verification steps in parallel using `Bash` tool with appropriate commands rather than waiting for each to complete sequentially.

### Best Practices for Parallel Task Execution with Subagents

When dividing work into parallel flows using subagents, follow these guidelines:

1. **Identify Independent Tasks**: Before launching subagents, analyze the work to identify tasks that can be executed independently without dependencies on each other's outputs.

2. **Use Descriptive Task Prompts**: When using the `task` tool, provide clear, detailed prompts that specify exactly what information the subagent should return, enabling effective parallel execution.

3. **Batch Subagent Launches**: Launch multiple subagents concurrently in a single message using multiple tool calls to maximize performance rather than launching them sequentially.

4. **Manage Task Coordination**: For tasks that require coordination after parallel execution, plan how the results will be integrated and what sequential steps (if any) need to follow.

5. **Monitor Resource Usage**: Be mindful of system resources when launching multiple subagents, particularly for resource-intensive tasks like large file processing or complex computations.

6. **Handle Failures Gracefully**: Design workflows that can handle partial failures in parallel tasks without compromising the entire process, using appropriate error handling and fallback strategies.
