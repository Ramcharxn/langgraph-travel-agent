# src/bedrock_client.py  (or src/llm/bedrock_client.py if that’s where you keep it)
import json
from langchain_aws import ChatBedrockConverse

base_llm = ChatBedrockConverse(
    model_id="openai.gpt-oss-120b-1:0",
    region_name="us-east-1",
    max_tokens=10000,
    temperature=0.7,
)


def _extract_json_from_text(raw: str) -> str:
    """
    Given a raw string that might contain reasoning content PLUS a JSON object,
    return just the JSON substring.

    Example input:
      "{'type': 'reasoning_content', ...}\n{ \"intent\": \"generic_chat\", ... }"

    Example output:
      "{ \"intent\": \"generic_chat\", ... }"
    """
    # Find the first '{' that starts a JSON-style object with double-quoted keys.
    # Simplest robust approach: find the first line that looks like JSON.
    # Split by lines and look for a line starting with '{' and containing '"intent"'.
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("{") and '"intent"' in line:
            json_str = "\n".join(lines[i:])
            return json_str

    # Fallback: if we can’t find it, just return the original string
    return raw


def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    tools: list | None = None,
) -> str:
    """
    Wrap Bedrock via LangChain ChatBedrockConverse.

    IMPORTANT: This returns a STRING that is expected to be valid JSON
    for your agents (master_agent does `json.loads(raw)`).
    """
    messages = [
        ("system", system_prompt),
        ("user", user_prompt),
    ]

    try:
        llm = base_llm.bind(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if tools:
            llm = llm.bind_tools(tools)

        ai_msg = llm.invoke(messages)

        raw_content = ai_msg.content

        # If it's already a string, great – just clean off any reasoning preamble.
        if isinstance(raw_content, str):
            cleaned = _extract_json_from_text(raw_content)
            return cleaned.strip()

        # If it's a list of blocks (new Converse format), pull out the TEXT blocks,
        # then extract JSON from the joined text.
        if isinstance(raw_content, list):
            text_chunks: list[str] = []
            for block in raw_content:
                if isinstance(block, dict):
                    # New reasoning models often use {"type": "reasoning_content", ...}
                    block_type = block.get("type")
                    if block_type == "output_text" or block_type == "text":
                        txt = block.get("text") or ""
                        if txt:
                            text_chunks.append(txt)
                    else:
                        # ignore reasoning/tool_use/etc
                        continue
                else:
                    text_chunks.append(str(block))

            joined = "\n".join(t for t in text_chunks if t)
            cleaned = _extract_json_from_text(joined)
            return cleaned.strip()

        # Fallback: stringify and try to strip reasoning/json
        cleaned = _extract_json_from_text(str(raw_content))
        return cleaned.strip()

    except Exception as e:
        print(f"Error calling Bedrock LLM: {e}")
        return ""
