"""
TOON (Tree Object Oriented Notation) Format Conversion
Converts JSON to a more compact format that uses fewer tokens
"""
import json
import re
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


def detect_json_in_prompt(prompt: str) -> Tuple[bool, str, int, int]:
    """
    Detect JSON in prompt and return (has_json, json_str, start_pos, end_pos)

    Args:
        prompt: The input prompt to search for JSON

    Returns:
        Tuple of (has_json, json_str, start_pos, end_pos)
    """
    try:
        # Try to find JSON objects
        json_start = prompt.find("{")
        json_end = prompt.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_str = prompt[json_start:json_end]
            # Validate it's actually JSON
            json.loads(json_str)
            return True, json_str, json_start, json_end

        # Also try arrays
        json_start = prompt.find("[")
        json_end = prompt.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            json_str = prompt[json_start:json_end]
            # Validate it's actually JSON
            json.loads(json_str)
            return True, json_str, json_start, json_end

    except (json.JSONDecodeError, ValueError):
        pass

    return False, "", -1, -1


def json_to_toon_basic(data: Any, indent: int = 0) -> str:
    """
    Basic TOON conversion - handles simple cases
    For production, you could use: pip install toon

    TOON is more compact than JSON:
    - Removes quotes around keys
    - Uses : for key-value pairs
    - Tabular format for uniform arrays
    - Removes unnecessary brackets where possible

    Args:
        data: Python dict/list to convert
        indent: Current indentation level

    Returns:
        TOON formatted string
    """
    spaces = "  " * indent

    if isinstance(data, dict):
        if not data:
            return f"{spaces}{{}}"

        result = []
        for k, v in data.items():
            if isinstance(v, (dict, list)) and v:
                result.append(f"{spaces}{k}:")
                result.append(json_to_toon_basic(v, indent + 1))
            else:
                # Simple value
                if isinstance(v, str):
                    result.append(f"{spaces}{k}: {v}")
                else:
                    result.append(f"{spaces}{k}: {v}")
        return "\n".join(result)

    elif isinstance(data, list):
        if not data:
            return f"{spaces}[]"

        # Check if uniform array of dicts (can use tabular format)
        if all(isinstance(x, dict) for x in data) and data:
            first_keys = set(data[0].keys())
            if all(set(x.keys()) == first_keys for x in data):
                # Tabular format - more compact
                fields = ",".join(first_keys)
                rows = []
                for item in data:
                    row_values = []
                    for k in first_keys:
                        val = item[k]
                        if isinstance(val, str):
                            row_values.append(f'"{val}"' if ',' in val else val)
                        else:
                            row_values.append(str(val))
                    rows.append(f"{spaces}  {','.join(row_values)}")
                header = f"{spaces}[{len(data)}]{{{fields}}}:"
                return header + "\n" + "\n".join(rows)

        # Non-uniform or simple array
        result = []
        for item in data:
            if isinstance(item, (dict, list)):
                result.append(json_to_toon_basic(item, indent + 1))
            else:
                result.append(f"{spaces}  {item}")
        return "\n".join(result)

    else:
        return f"{spaces}{data}"


def convert_prompt_to_toon(prompt: str) -> Tuple[str, int]:
    """
    Convert JSON in prompt to TOON format

    Args:
        prompt: Input prompt potentially containing JSON

    Returns:
        Tuple of (converted_prompt, estimated_tokens_saved)
    """
    has_json, json_str, start, end = detect_json_in_prompt(prompt)

    if not has_json:
        logger.debug("No JSON detected in prompt")
        return prompt, 0

    try:
        json_obj = json.loads(json_str)
        toon_str = json_to_toon_basic(json_obj)

        # Replace in prompt
        new_prompt = prompt[:start] + toon_str + prompt[end:]

        # Estimate tokens saved (rough estimate: character difference / 4)
        # This is approximate - actual savings will be measured by API
        chars_saved = len(json_str) - len(toon_str)
        tokens_saved = max(0, chars_saved // 4)

        logger.info(
            f"TOON conversion: {len(json_str)} chars -> {len(toon_str)} chars "
            f"(~{tokens_saved} tokens saved estimate)"
        )

        return new_prompt, tokens_saved

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to convert JSON to TOON: {str(e)}")
        return prompt, 0


def add_json_to_prompt(prompt: str, json_data: Any) -> str:
    """
    Add JSON data to a prompt

    Args:
        prompt: Base prompt
        json_data: Data to add (dict or list)

    Returns:
        Prompt with JSON appended
    """
    if isinstance(json_data, str):
        # Already a string, try to parse it
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            # Not valid JSON, just append as-is
            return f"{prompt}\n\n{json_data}"

    json_str = json.dumps(json_data, indent=2)
    return f"{prompt}\n\n{json_str}"
