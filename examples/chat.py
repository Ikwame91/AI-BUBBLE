"""A complete terminal chatbot in ~30 lines.

The key idea: the model has no memory. This script keeps the conversation in
`history` and re-sends all of it on every turn.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python chat.py
"""
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
history = []                    # the model's only "memory"

SYSTEM = "You are a patient tutor who explains AI concepts simply."

while True:
    user = input("you > ").strip()
    if user in {"quit", "exit"}:
        break
    history.append({"role": "user", "content": user})

    reply = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=4096,
        system=SYSTEM,
        messages=history,                       # the WHOLE conversation, every turn
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                    # if a safety classifier declines, retry on a fallback model
    )

    if reply.stop_reason == "refusal":
        print("claude > (declined this request; try rephrasing)")
        history.pop()                           # drop the message that was declined
        continue

    text = "".join(b.text for b in reply.content if b.type == "text")
    history.append({"role": "assistant", "content": reply.content})
    print("claude >", text)
    print(f"[{reply.usage.input_tokens} tokens in, {reply.usage.output_tokens} out]")
