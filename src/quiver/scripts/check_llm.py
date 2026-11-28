from pathlib import Path

from dotenv import load_dotenv

from quiver.llm_selector import get_catalog_dict, select_tool, create_client

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

if __name__ == "__main__":
    load_dotenv()
    client = create_client()
    catalog = get_catalog_dict(CATALOG_PATH)
    selection = select_tool("fly with plane", 0.2, catalog, client)
    for tool_choice in selection.tools:
        print(tool_choice.tool + " ----- " + tool_choice.reason)
