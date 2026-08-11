> **Historical document.** This describes an earlier coding-operator concept (CLI, five pillars, Rust/SQLite SM). It is **not** the Operator Architecture SDK contract. See [README.md](README.md) for the current orchestration SDK.

# Operator Architecture Technical Specification

## Overview
The Operator Architecture is designed to solve the problem of "context explosion" in large-scale LLM-driven development. Instead of a single behemoth implementation that loses accuracy as context windows fill, this architecture splits complex tasks into focused, purpose-built **Sub-Agents**. Each sub-agent operates with hyper-efficient context management, receiving only the specific information required to complete its current objective, while a centralized state machine maintains the global truth.

## Core Components

### 1. The Coordinator
The primary agent persona for user interaction. The Coordinator acts as the orchestrator and reviewer, translating high-level user intent into actionable objectives and validating the outputs of sub-agents. It is the sole interface for the human user.

### 2. The State Machine
The backbone of the entire system. Implemented in **Rust**, this layer manages orchestration and persistence via **SQLite**.
- **Session Management**: Tracks active agent sessions.
- **Plan & State**: Maintains the current plan, step statuses, and objective progress.
- **Audit Log**: Tracks all file edits (diff logs) for traceability and safety.

### 3. Sub-Agents (The 5 Pillars)
Each sub-agent is specialized in a specific domain to minimize token noise and maximize reasoning capabilities:
- **Researcher**: A read-only toolkit specialized in codebase navigation, grep, and semantic comprehension.
- **Implementer**: A read/write toolkit capable of performing complex code modifications and surgical edits.
- **Bash/Tester**: A terminal-access agent designed to execute commands, run test suites, and verify environment state.
- **Documenter**: A specialized toolkit for generating, updating, and maintaining Markdown-based documentation.
- **Planner**: A strategic agent responsible for breaking down high-level objectives into granular, executable implementation steps.

## The Workflow

### Commissioning
The Coordinator does not simply "talk" to agents; it commissions them using the State Machine. It passes structured properties to sub-agents:
- **Objective**: The specific goal of the sub-agent.
- **Checklist**: A list of requirements to be met.
- **Agent Props**: Context-specific metadata needed for the task.

### Plan Execution
Plans are executed incrementally:
1. The **Planner** creates a sequence of steps.
2. **Implementation Loop**. Each step triggers an implementation agent with the context of the plan overview, all the steps overview, it's specific step implementation plan, and the agent summary from previously completed steps.
3. **Verification Loop**: Implementation steps can trigger the **Bash/Tester** agent automatically to verify that the code change satisfies the requirement before the step is marked as complete.

### Context Efficiency
To prevent hallucinations and token waste, agents do not receive full conversation histories. They receive:
- A high-level **Plan Overview**.
- The current **Step Objective**.
- **Summaries** of prior successful steps.
This ensures the LLM's attention is focused entirely on the task at hand.

## The CLI & Server Model

### Lifecycle Management
The `operator` CLI manages a persistent background server:
- `operator server --start`: Starts the orchestration server.
- `operator server --stop`: Gracefully shuts down the server.
- `operator server --bind <session>`: Attaches the CLI to an existing session/project.
- `operator server --drop <session>`: Detaches the live session from the project.

### Intelligent Automation
The CLI is designed for ease of use: if a command is issued while the server is offline, the CLI automatically handles the start/bind sequence.

### Extensibility
The system exposes an **OpenAPI surface on port 9200**, allowing web-based dashboards or third-party tools to orchestrate agents and monitor the State Machine in real-time.

This allows developers to integrate into IDEs or create their own visual agent windows that utilize the Operator Architecture.

## Technical Stack
- Python
- **Persistence**: [SQLite](https://www.sqlite.org/) (Lightweight, file-based, and ACID-compliant for session and diff tracking).
- **API**: REST/OpenAPI (For seamless integration). Using **FastAPI**
- **Configuration**: Managed via a `.operator` directory containing the database and system settings.

## Future Work
- **Parallel Step Execution**: Allowing multiple non-dependent sub-agents to work simultaneously.
- **Courier Cloud**: Integration for remote authentication and automated configuration population.

