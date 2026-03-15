#!/usr/bin/env python3
"""
Chat with an Ollama model that uses the data-profiler MCP server as tools.

Usage:
    python3 run_with_ollama.py [model_name] [csv_path]

Defaults:
    model_name = llama3.1:8b
    csv_path   = test_data/simple.csv
"""

import json
import os
import subprocess
import sys

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_BIN = os.path.join(SCRIPT_DIR, "target", "release", "mcp")
OLLAMA_URL = "http://localhost:11434/api/chat"


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
# Convert MCP tool schemas → Ollama tool format
# ---------------------------------------------------------------------------

def mcp_tools_to_ollama(mcp_tools: list[dict]) -> list[dict]:
    ollama_tools = []
    for t in mcp_tools:
        schema = t.get("inputSchema", {})
        ollama_tools.append({
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
    return ollama_tools


# ---------------------------------------------------------------------------
# Ollama chat with tool-calling loop
# ---------------------------------------------------------------------------

def chat_with_tools(model: str, messages: list[dict], tools: list[dict],
                    mcp: McpClient, max_rounds: int = 20) -> str:
    for _ in range(max_rounds):
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        })
        resp.raise_for_status()
        data = resp.json()
        msg = data["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # No more tool calls — model is done.
            return msg.get("content", "")

        # Execute each tool call via MCP.
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            print(f"  🔧 Calling tool: {name}({json.dumps(args)})")
            result = mcp.call_tool(name, args)
            print(f"  ← {result[:200]}{'...' if len(result) > 200 else ''}")
            messages.append({"role": "tool", "content": result})

    return "Reached maximum tool-calling rounds."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.1:8b"
    csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRIPT_DIR, "test_data", "simple.csv")
    csv_path = os.path.abspath(csv_path)

    print(f"Model:   {model}")
    print(f"Dataset: {csv_path}")
    print()

    # Build if needed.
    if not os.path.exists(MCP_BIN):
        print("Building MCP server...")
        subprocess.run(["cargo", "build", "--release"], cwd=SCRIPT_DIR, check=True)

    # Start MCP server.
    mcp = McpClient(MCP_BIN)
    mcp.initialize()
    mcp_tools = mcp.list_tools()
    ollama_tools = mcp_tools_to_ollama(mcp_tools)
    print(f"MCP server started — {len(mcp_tools)} tools available.\n")

    # Load the dataset first.
    print(f"Loading dataset: {csv_path}")
    result = mcp.call_tool("load_dataset", {"path": csv_path})
    print(f"  ← {result}\n")

    # Chat loop.
    system_prompt = (
        "You are a data analyst. A CSV dataset has already been loaded into the "
        "data-profiler tool. Use the available tools to explore and analyze the "
        "dataset. Start by examining its shape and column types, then provide a "
        "thorough statistical summary. For any numeric columns that look like "
        "time series, also run the reservoir computing diagnostics (surrogate_test, "
        "bds_test, lyapunov_exponent, dependence_comparison, delay_embedding, "
        "memory_profile)."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Profile this dataset thoroughly. Give me a complete analysis."},
    ]

    print("=" * 60)
    print("Sending to Ollama...")
    print("=" * 60)
    print()

    answer = chat_with_tools(model, messages, ollama_tools, mcp)

    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(answer)

    mcp.close()


if __name__ == "__main__":
    main()
