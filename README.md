---
title: AgenticAI
emoji: 🚀
colorFrom: indigo
colorTo: yellow
sdk: streamlit
sdk_version: 1.49.1
app_file: app.py
pinned: false
license: mit
short_description: 'PDF Chat with AgenticAI. '
---

# Agentic AI 🤖

## 📌 Overview
This project is a **agentic AI framework** with hybrid RAG system. It demonstrates how to build an LLM-powered agent that can:
- **Reason** step by step (Plan → Act → Reflect → Answer).
- **Use tools** (calculator, RAG, internet search, summarization, memory lookup).
- **Store and retrieve knowledge** with long-term memory.
- **Enforce guardrails** so the agent stays grounded to documents.

It is designed to be simple, extensible, and easy to plug into applications (e.g., Streamlit chatbots).

Accesible in Hugging Face space - [https://huggingface.co/spaces/polojuan/agenticAI](https://huggingface.co/spaces/polojuan/agenticAI)

---

## 🏗️ Architecture


### Components
1. **LLM Adapter**
- Wraps OpenAI (or another provider).
- Handles completions, token limits, and temperature.
- Can be swapped with Anthropic, local models, etc.


2. **Agent Core**
   - The reasoning loop that implements the **ReAct pattern**:
     1. **Thought** → plan next step.
     2. **Action** → decide which tool to call.
     3. **Observation** → receive tool output.
     4. **Reflection** → critique correctness.
     5. Repeat until **Final Answer**.


3. **Tools Registry**
   - Each tool has a name, description, JSON schema, and a Python function.
   - Tools include:
   - **rag** → retrieval-augmented generation
   - **internet** → limited web search
   - **search_memory / write_memory** → semantic memory access
   - You can add any other tool required


4. **Memory**
   - JSONL file for long-term notes.
   - TF-IDF-lite search for retrieval.
   - Stores goals, tool calls, answers, and notes.


5. **Critic Module**
   - After each step, a “critic” pass reviews correctness and safety.
   - Detects tool misuse, hallucinations, or off-topic answers.
   - Suggests corrections.


---


## 🔄 Agentic Workflow


Here’s the **step-by-step workflow**:


1. **User Goal**: The user submits a query (e.g., *“Summarize this article”*).


2. **Context Building**:
   - Agent retrieves relevant memory.
   - Injects available tool manifest.
   - Builds a structured system prompt with rules.


3. **LLM Reasoning Loop**:
   - LLM outputs:
   - `Thought: …` → internal reasoning.
   - `Action: <tool> {args}` → request to use a tool.
   - `Final: …` → final answer.


4. **Tool Execution**:
   - Python validates arguments against schema.
   - Executes the tool (e.g., call `tool_rag`).
   - Returns `Observation[...]` with results.


5. **Reflection**:
   - A critic LLM checks if the last step is correct/safe.
   - Flags errors (e.g., wrong tool, hallucination).
   - Suggests adjustments.


6. **Loop Control**:
   - Continues until:
   - `Final:` is reached, OR
   - Max steps/tokens hit → fallback answer.


7. **Answer Storage**:
   - Final answer stored in memory with metadata.
   - Can be retrieved in future queries.

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+  
- Docker (if using containerized mode)  
- Access to a language model (OpenAI API key, local LLM, etc.)  

### Local Setup (without Docker)

```bash
git clone https://github.com/palscruz23/agenticAI.git
cd agenticAI

pip install -r requirements.txt

# Run the app (assuming it’s a Streamlit or web frontend)
streamlit run app.py
```
---

## 📜 License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.