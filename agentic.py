"""
Agentic AI Starter – OpenAI + Step-by-step Logging/Tracing
Single-file, production‑lean demo with:
- ReAct loop (plan → act → observe → reflect → answer)
- Tools w/ JSON‑schema validation & timeouts
- Simple long‑term memory (JSONL + tf‑idf‑ish)
- OpenAI Chat Completions client (>=1.0 SDK)
- Built‑in step‑by‑step console logs + structured JSON trace
"""
from __future__ import annotations
import json, os, time, math, re, textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dotenv import load_dotenv
from utils.hybrid import hybrid_rag_bm25
import requests

# Load environment variables
load_dotenv(".env")

# OpenAI SDK (>=1.0)
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # allow mock usage if SDK unavailable

# ===============================
# JSON Schema (minimal)
# ===============================
class SchemaError(Exception):
    pass

def isinstance_type(x: Any, t: str) -> bool:
    mapping = {
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)),
        "integer": lambda v: isinstance(v, int),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    return mapping.get(t, lambda v: True)(x)

def validate_json(data: Any, schema: Dict[str, Any]):
    if schema.get("type") != "object":
        raise SchemaError("Only object schemas supported")
    if not isinstance(data, dict):
        raise SchemaError("Expected object")
    required = schema.get("required", [])
    for r in required:
        if r not in data:
            raise SchemaError(f"Missing required field: {r}")
    props = schema.get("properties", {})
    for k, v in data.items():
        if k not in props:
            raise SchemaError(f"Unexpected field: {k}")
        expect = props[k].get("type")
        if expect and not isinstance_type(data[k], expect):
            raise SchemaError(f"Field {k} should be {expect}")

# ===============================
# Utilities
# ===============================

def run_with_timeout(fn: Callable, args: Dict[str, Any], timeout: float):
    start = time.time()
    result = fn(args)
    if time.time() - start > timeout:
        raise TimeoutError("Tool execution exceeded timeout")
    return result


def tok(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())


def safe_truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."

# ===============================
# LLM adapters
# ===============================
class BaseLLM:
    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        raise NotImplementedError

class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise RuntimeError("openai SDK not available. pip install openai>=1.0.0")
        self.model = model
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])  # reads OPENAI_API_KEY from env

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

# Optional mock for offline testing
class MockLLM(BaseLLM):
    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 5000) -> str:
        lines = [l for l in prompt.splitlines() if l.strip()]
        # Very basic behavior: if a tool is available, use calculator when numbers present
        want_tool = any(ch.isdigit() for ch in "".join(lines[-10:]))
        if want_tool:
            return "Thought: I should compute this with a tool.\nAction: calculator {\"expression\": \"(23*7+9)/4\"}"
        return "Thought: I can answer now.\nFinal: This is a mock answer."

# ===============================
# Tools
# ===============================
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], Any]
    timeout_s: float = 15.0

    def __call__(self, args: Dict[str, Any]) -> Any:
        validate_json(args, self.schema)
        return run_with_timeout(self.func, args, timeout=self.timeout_s)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"No such tool: {name}")
        return self._tools[name]

    def manifest(self) -> List[Dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "schema": t.schema}
            for t in self._tools.values()
        ]

# Memory tools

def tool_search_memory(args: Dict[str, Any], mem: "Memory") -> Dict[str, Any]:
    q = args["query"]
    k = int(args.get("k", 3))
    hits = mem.search(q, k=k)
    return {"hits": hits}

def tool_write_memory(args: Dict[str, Any], mem: "Memory") -> Dict[str, Any]:
    content = args["content"]
    mem.add("note", content, {"source": "tool"})
    return {"ok": True}

# RAG tool
def tool_rag(args: Dict[str, Any]) -> Dict[str, Any]:
    if hybrid_rag_bm25 is None:
        raise RuntimeError("query_rag helper not available")
    query = args["query"]
    formatted_response, _ = hybrid_rag_bm25(query)
    return {"result": formatted_response}


def tool_internet(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple web search tool using DuckDuckGo Instant Answer API.
    Input: {"query": "what is the capital of Japan?"}
    Output: {"result": "Tokyo is the capital of Japan."}
    """
    query = args["query"]
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        data = resp.json()
        answer = data.get("AbstractText") or data.get("Heading") or "No direct answer found."
        return {"result": answer}
    except Exception as e:
        return {"error": str(e)}

# ===============================
# Memory (JSONL + tf-idf-ish)
# ===============================
@dataclass
class Memory:
    path: str = "memory.jsonl"
    _cache: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._cache = [json.loads(line) for line in f if line.strip()]

    def add(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None):
        rec = {"role": role, "content": content, "meta": meta or {}, "ts": time.time()}
        self._cache.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        df = defaultdict(int)
        for rec in self._cache:
            for t in set(tok(rec["content"])):
                df[t] += 1
        scores: List[Tuple[float, Dict[str, Any]]] = []
        q = set(tok(query))
        for rec in self._cache:
            w = 0.0
            for t in q.intersection(set(tok(rec["content"]))):
                idf = math.log(1 + len(self._cache) / (1 + df[t]))
                w += idf
            if w > 0:
                scores.append((w, rec))
        return [r for _, r in sorted(scores, key=lambda x: -x[0])[:k]]

# ===============================
# Agent core
# ===============================
@dataclass
class AgentConfig:
    answer_depth: str = "detailed"  # options: concise, detailed, verbose
    max_steps: int = 8
    temperature: float = 0.2
    reflection: bool = True
    debug: bool = True               # print step-by-step logs to console
    max_obs_chars: int = 1500        # truncate noisy observations

@dataclass
class Agent:
    llm: BaseLLM
    tools: ToolRegistry
    memory: Memory
    config: AgentConfig = field(default_factory=AgentConfig)

    def run(self, user_goal: str, *, debug: Optional[bool] = None, return_trace: bool = False) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        debug = self.config.debug if debug is None else debug
        trace: List[Dict[str, Any]] = []
        transcript: List[Dict[str, str]] = []
        self.memory.add("user_goal", user_goal)

        def plog(msg: str):
            if debug:
                print(msg)

        for step in range(1, self.config.max_steps + 1):
            context = self._build_context(user_goal, transcript)
            thought = self.llm.complete(context, temperature=self.config.temperature)
            transcript.append({"role":"assistant","content": thought})

            thought_line = _extract_first_line(thought, prefix="Thought:")
            plog(f"Step {step} Thought: {thought_line}")

            action = parse_action(thought)
            obs = None
            critique_text = None

            if action["type"] == "final":
                answer = action["content"].strip()
                plog(f"Step {step} Final: {safe_truncate(answer, 200)}")
                self.memory.add("final_answer", answer, {"steps": step})
                trace.append({
                    "step": step,
                    "thought": thought_line,
                    "action": {"type": "final", "content": answer},
                    "observation": None,
                    "critique": None,
                })
                return (answer, trace) if return_trace else answer

            if action["type"] == "tool":
                tool_name = action["tool"]
                args = action["args"]
                plog(f"Step {step} Action: {tool_name} {args}")
                try:
                    tool = self.tools.get(tool_name)
                    result = tool(args)
                    obs = f"Observation[{tool_name}]: {safe_truncate(str(result), self.config.max_obs_chars)}"
                except Exception as e:
                    obs = f"Observation[{tool_name} ERROR]: {type(e).__name__}: {e}"
                transcript.append({"role":"tool","content": obs})
                self.memory.add("tool_use", obs)
                plog(f"Step {step} {safe_truncate(obs, 200)}")

            if self.config.reflection:
                critique_prompt = make_critique_prompt(user_goal, transcript)
                critique = self.llm.complete(critique_prompt, temperature=0.1)
                transcript.append({"role":"critic","content": critique})
                critique_text = _extract_first_line(critique, prefix="Critique:")
                plog(f"Step {step} Critique: {critique_text}")

            trace.append({
                "step": step,
                "thought": thought_line,
                "action": action,
                "observation": obs,
                "critique": critique_text,
            })

        fallback = "I reached my step limit. Best effort summary: "
        best = infer_best_answer(transcript) or "No definitive answer achieved."
        final = fallback + best
        self.memory.add("final_answer", final, {"steps": "limit"})
        plog(f"Final (limit): {safe_truncate(final, 200)}")
        return (final, trace) if return_trace else final

    def _build_context(self, user_goal: str, transcript: List[Dict[str, str]]) -> str:
        retrieved = self.memory.search(user_goal, k=3)
        tool_manifest = json.dumps(self.tools.manifest(), ensure_ascii=False)
        if self.config.answer_depth == "concise":
            style = "short and concise"
        elif self.config.answer_depth == "detailed":
            style = "well-structured and detailed (1-3 paragraphs)"
        else:
            style = "very detailed, elaborate, and multi-paragraph"
        history = "\n".join(f"[{m['role'].upper()}] {m['content']}" for m in transcript[-8:])
        return f"""
You are an agent that Plans, Acts (via tools), Reflects, and Answers.
Follow this strict output format on each step:
- Start with a short "Thought: ..." line explaining your next best action.
- If using a tool, output a single line: Action: <tool_name> <JSON-args>
- If ready to answer, output:
  Final: <your best {style} answer>
  (Your Final answer can be multi-line and detailed. If the task is open-ended, provide a thorough response.)

User goal: {user_goal}

Available tools (JSON schemas):
{tool_manifest}

Relevant memory:
{json.dumps(retrieved, ensure_ascii=False)}

Recent transcript:
{history}
""".strip()

# ===============================
# Parsing & critique helpers
# ===============================

def parse_action(text: str) -> Dict[str, Any]:
    """Parse model output for Final or Action. Supports multi-line answers/JSON."""
    # Final: capture everything after "Final:" to the end (multi-line)
    m = re.search(r"^Final:\s*(.*)", text, re.MULTILINE | re.DOTALL)
    if m:
        return {"type": "final", "content": m.group(1).strip()}

    # Action: capture tool name + JSON args (allow multi-line JSON; non-greedy)
    m = re.search(r"^Action:\s*([a-zA-Z0-9_\-]+)\s*(\{.*?\})", text, re.MULTILINE | re.DOTALL)
    if m:
        tool = m.group(1)
        json_str = m.group(2).strip()
        try:
            args = json.loads(json_str)
        except Exception:
            # last-resort: try swapping single→double quotes
            try:
                args = json.loads(re.sub(r"'", '"', json_str))
            except Exception:
                args = {}
        return {"type": "tool", "tool": tool, "args": args}

    return {"type": "none", "content": text}



def _extract_first_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()[len(prefix):].strip()
    return safe_truncate(text.replace("\n", " "), 160)


def make_critique_prompt(user_goal: str, transcript: List[Dict[str,str]]) -> str:
    last = transcript[-1]["content"] if transcript else ""
    return f"""
Critic: Evaluate the previous step for correctness and safety with respect to the goal: {user_goal}.
Rules: If the step is off-topic, requests private data, or misuses a tool, state it and suggest a fix in ONE short sentence.
Respond with: Critique: <one short sentence>. If everything is fine, say: Critique: OK.

Previous step:
{last}
""".strip()


def infer_best_answer(transcript: List[Dict[str,str]]) -> str:
    finals = [m["content"][7:] for m in transcript if m["content"].startswith("Final:")]
    if finals:
        return finals[-1]
    obs = next((m["content"] for m in reversed(transcript) if m["role"]=="tool"), "")
    return obs[:400] if obs else ""

# ===============================
# Bootstrap
# ===============================

def bootstrap_example(use_openai: bool = True) -> Agent:
    tools = ToolRegistry()
    memory = Memory()

    
    tools.register(Tool(
        name="rag",
        description="Retrieves semantic context and keyword search from the knowledge base.",
        schema={
            "type":"object",
            "properties": {"content": {"type":"string"}},
            "required":["content"],
        },
        func=tool_rag,
    ))

    tools.register(Tool(
        name="search_memory",
        description="Semantic-ish search over long-term memory store.",
        schema={
            "type":"object",
            "properties": {"query": {"type":"string"}, "k": {"type":"integer"}},
            "required":["query"],
        },
        func=lambda args, mem=memory: tool_search_memory(args, mem),
    ))

    tools.register(Tool(
        name="write_memory",
        description="Write a short note to long-term memory for future retrieval.",
        schema={
            "type":"object",
            "properties": {"content": {"type":"string"}},
            "required":["content"],
        },
        func=lambda args, mem=memory: tool_write_memory(args, mem),
    ))

    tools.register(Tool(
        name="internet",
        description="Look up real-time information from the web. Useful for current events and facts.",
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        func=tool_internet,
    ))

    llm: BaseLLM = OpenAILLM() if (use_openai and OpenAI is not None) else MockLLM()

    agent = Agent(
        llm=llm,
        tools=tools,
        memory=memory,
        config=AgentConfig(max_steps=20, reflection=True, debug=True),
    )
    return agent

# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    use_openai = bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None
    agent = bootstrap_example(use_openai=use_openai)
    print("Agentic AI Starter – demo. Type a goal, or 'exit'. Logging is ON.\n")
    while True:
        try:
            goal = input("> ").strip()
            if not goal:
                continue
            if goal.lower() in {"exit","quit"}:
                break
            print("\n--- Running agent ---")
            answer, trace = agent.run(goal, return_trace=True)  # always return trace in CLI
            print("\n=== Trace (JSON) ===")
            print(json.dumps(trace, indent=2, ensure_ascii=False))
            print()
            print(json.dumps(trace, indent=2, ensure_ascii=False))
            print("\n=== Final Answer ===\n" + str(answer))

        except KeyboardInterrupt:
            print("\nbye\n"); break
