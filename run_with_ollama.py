#!/usr/bin/env python3
"""
Chat with an OpenAI model that uses the data-profiler MCP server as tools.

Usage:
    python3 run_with_ollama.py <csv_path> [model_name]

Requires:
    OPENAI_API_KEY environment variable

Defaults:
    model_name = gpt-4o
"""

import json
import os
import subprocess
import sys

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_BIN = os.path.join(SCRIPT_DIR, "target", "release", "mcp")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# MCP client — thin wrapper around the subprocess
# ---------------------------------------------------------------------------

class McpClient:
    def __init__(self, bin_path: str):
        self.proc = subprocess.Popen(
            [bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def send(self, method: str, params: dict | None = None) -> dict:
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode())
        self.proc.stdin.flush()
        resp_line = self.proc.stdout.readline().decode()
        return json.loads(resp_line)

    def initialize(self) -> dict:
        return self.send("initialize")

    def list_tools(self) -> list[dict]:
        resp = self.send("tools/list")
        return resp["result"]["tools"]

    def call_tool(self, name: str, arguments: dict) -> str:
        resp = self.send("tools/call", {"name": name, "arguments": arguments})
        result = resp.get("result", {})
        content = result.get("content", [])
        if content:
            return content[0].get("text", json.dumps(result))
        return json.dumps(result)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------------------
# Convert MCP tool schemas → OpenAI tool format
# ---------------------------------------------------------------------------

def mcp_tools_to_openai(mcp_tools: list[dict]) -> list[dict]:
    openai_tools = []
    for t in mcp_tools:
        schema = t.get("inputSchema", {})
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": {
                    "type": schema.get("type", "object"),
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        })
    return openai_tools


# ---------------------------------------------------------------------------
# OpenAI chat with tool-calling loop
# ---------------------------------------------------------------------------

def chat_with_tools(model: str, api_key: str, messages: list[dict],
                    tools: list[dict], mcp: McpClient,
                    max_rounds: int = 20) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for round_num in range(max_rounds):
        try:
            resp = requests.post(OPENAI_URL, headers=headers, json={
                "model": model,
                "messages": messages,
                "tools": tools,
            }, timeout=300)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            print("  ⚠ OpenAI request timed out, finishing with partial results.")
            return "Analysis incomplete — request timed out."
        except requests.exceptions.HTTPError as e:
            print(f"  ⚠ OpenAI API error: {e}")
            print(f"    {resp.text[:500]}")
            return f"Analysis incomplete — API error: {e}"

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        # Append assistant message to conversation.
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "")

        # Execute each tool call via MCP.
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            call_id = tc["id"]
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            # Skip if model tries to load_dataset again.
            if name == "load_dataset":
                print(f"  ⏭ Skipping redundant load_dataset call")
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": "Dataset is already loaded."})
                continue

            print(f"  🔧 Calling tool: {name}({json.dumps(args)})")
            result = mcp.call_tool(name, args)
            print(f"  ← {result[:200]}{'...' if len(result) > 200 else ''}")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    return "Reached maximum tool-calling rounds."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_with_ollama.py <csv_path> [model_name]", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    csv_path = os.path.abspath(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1"

    if not os.path.isfile(csv_path):
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model:   {model}")
    print(f"Dataset: {csv_path}")
    print()

    # Build if needed.
    if not os.path.exists(MCP_BIN):
        print("Building MCP server...")
        subprocess.run(["cargo", "build", "--release"], cwd=SCRIPT_DIR, check=True)

    name = os.path.basename(csv_path)

    print()
    print("#" * 60)
    print(f"# Dataset: {name}")
    print("#" * 60)
    print()

    # Start MCP server.
    mcp = McpClient(MCP_BIN)
    mcp.initialize()
    mcp_tools = mcp.list_tools()
    # Remove load_dataset from tools sent to model — we already loaded it.
    openai_tools = [t for t in mcp_tools_to_openai(mcp_tools) if t["function"]["name"] != "load_dataset"]
    print(f"MCP server started — {len(openai_tools)} tools available.")

    # Load the dataset.
    print(f"Loading dataset: {csv_path}")
    result = mcp.call_tool("load_dataset", {"path": csv_path})
    print(f"  ← {result}\n")

    load_info = json.loads(result)
    rows = load_info.get('rows', '?')
    cols = load_info.get('columns', '?')

    system_prompt = (
        f"You are a thorough data analyst. The CSV dataset '{name}' is already loaded "
        f"({rows} rows, {cols} columns). Do NOT call load_dataset — it is already done.\n\n"
        "IMPORTANT: You MUST call ALL of the following tools before writing ANY analysis. "
        "Do NOT write a final answer until every step below is complete. "
        "Do NOT summarize or explain your plan — just call tools.\n\n"
        "Step 1: Call column_types to discover the columns.\n"
        "Step 2: Call correlation_matrix.\n"
        "Step 3: For EVERY numeric column, call each of these tools (one call per column):\n"
        "   mean, variance, quantiles, skewness, entropy, sparsity\n"
        "Step 4: For EVERY numeric column, call each of these tools (one call per column):\n"
        "   surrogate_test, bds_test (use embedding_dim=3, epsilon=1.0), "
        "lyapunov_exponent, dependence_comparison, delay_embedding, memory_profile\n\n"
        "You may batch multiple tool calls in a single turn. "
        "Keep calling tools until you have results from ALL tools for ALL columns. "
        "Only after ALL tool calls are done, write a comprehensive analysis that covers:\n"
        "- Summary statistics (mean, variance, quantiles, skewness) for each column\n"
        "- Correlations between columns\n"
        "- Entropy and sparsity patterns\n"
        "- Nonlinearity findings: interpret surrogate test z-scores, BDS test p-values, "
        "Lyapunov exponents, and whether nonlinear dependence dominates linear dependence\n"
        "- What these findings reveal about the data-generating process\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Analyze '{name}' thoroughly. Begin by calling column_types, then proceed "
            "through ALL the steps. Call every tool for every column — do not skip any. "
            "Do not write your final analysis until you have called all tools."
        )},
    ]

    print("=" * 60)
    print("Sending to OpenAI...")
    print("=" * 60)
    print()

    answer = chat_with_tools(model, api_key, messages, openai_tools, mcp)

    print()
    print("=" * 60)
    print(f"ANALYSIS — {name}")
    print("=" * 60)
    print(answer)

    mcp.close()


if __name__ == "__main__":
    main()
