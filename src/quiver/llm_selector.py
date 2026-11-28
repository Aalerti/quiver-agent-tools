import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError
from quiver.models import Selection

from quiver.exceptions.selection_error import SelectionError


def select_tool(task: str, temperature: float, catalog: dict[str, str], client: OpenAI) -> Selection:
    response = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=temperature,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You help select tools."
                                          " Select the tools needed throughout the entire "
                                          "chain of work, not only the ones needed at the first step. "
                                          "Select tools strictly from the list below and do not add extra ones."
                                          " List format: "
                                          "tool name - short description of when to use it. "
                                          "If there are no suitable tools, return an empty list.\n"
                                          + format_catalog(catalog)
             },
            {"role": "user", "content": task},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "selection",
                "schema": Selection.model_json_schema()
            },
        }
    )

    choice = response.choices[0]

    if choice.finish_reason == "length": raise SelectionError("Model ran out of output tokens")
    if not choice.message.content: raise SelectionError("Model didn't output anything")
    try:
        selection = Selection.model_validate_json(choice.message.content)
    except ValidationError as e:
        raise SelectionError(f"Invalid model output: {e}")
    validate_tool_name(selection, catalog)

    return selection


def get_catalog_dict(path: Path) -> dict[str, str]:
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    if not data: raise SelectionError("No tools in catalog json")

    catalog = {}
    for d in data:
        catalog[d["name"]] = d["description"]

    return catalog


def format_catalog(catalog: dict[str, str]) -> str:
    return '\n'.join(name + " - " + description for name, description in catalog.items())


def validate_tool_name(selection: Selection, catalog: dict[str, str]) -> None:
    for tool_choice in selection.tools:
        if tool_choice.tool not in catalog: raise SelectionError(f"Tool {tool_choice.tool} not in catalog")


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        project=os.environ["LLM_FOLDER_ID"],
    )
