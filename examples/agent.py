"""The agent loop, written out by hand so every step is visible.

1. Send the conversation plus a list of tools the model may call.
2. If the model answers with `tool_use`, YOUR code runs the tool...
3. ...and sends the result back as a `tool_result`.
4. Repeat until the model stops asking for tools (stop_reason == "end_turn").

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python agent.py "Do I need an umbrella in Lisbon tomorrow?"
"""
import json
import sys

import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_forecast",
        "description": "Get the weather forecast for a city. Returns rain chance (0-1) and high temperature in Celsius.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Lisbon"},
                "day": {"type": "string", "enum": ["today", "tomorrow"]},
            },
            "required": ["city", "day"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def get_forecast(city: str, day: str) -> dict:
    # A real app would call a weather API here. Fixed data keeps the example offline.
    return {"city": city, "day": day, "rain_chance": 0.7, "high_c": 17}


def run_tool(name: str, args: dict) -> str:
    if name == "get_forecast":
        return json.dumps(get_forecast(**args))
    raise ValueError(f"unknown tool {name}")


question = " ".join(sys.argv[1:]) or "Do I need an umbrella in Lisbon tomorrow?"
messages = [{"role": "user", "content": question}]

for turn in range(10):  # a safety cap on the number of loop iterations
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        tools=TOOLS,
        messages=messages,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    print(f"\n--- model turn {turn + 1}: stop_reason={response.stop_reason}")

    if response.stop_reason == "refusal":
        print("The request was declined.")
        break

    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        print("".join(b.text for b in response.content if b.type == "text"))
        break

    results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"model asked to call {block.name}({json.dumps(block.input)})")
            try:
                output, is_error = run_tool(block.name, block.input), False
            except Exception as e:  # report tool failures back to the model instead of crashing
                output, is_error = str(e), True
            print(f"tool returned {output}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
    # All results for one turn go back together, in a single user message.
    messages.append({"role": "user", "content": results})
