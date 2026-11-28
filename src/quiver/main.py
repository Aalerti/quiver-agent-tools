import os
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from quiver.exceptions.selection_error import SelectionError
from quiver.llm_selector import select_tool, create_client, get_catalog_dict
from quiver.models import SelectionRequest, Selection

DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    catalog_path = Path(os.getenv("CATALOG_PATH") or DEFAULT_CATALOG_PATH)
    app.state.client = create_client()
    app.state.catalog = get_catalog_dict(catalog_path)
    yield


app = FastAPI(title="Quiver", lifespan=lifespan)


@app.post("/select-tools")
def get_select_tool(body: SelectionRequest, request: Request) -> Selection:
    try:
        selection = select_tool(
            body.task,
            body.temperature,
            request.app.state.catalog,
            request.app.state.client
        )
    except (SelectionError, openai.APIError) as e:
        raise HTTPException(status_code=502, detail="LLM providers error") from e
    return selection

@app.get("/health")
def health():
    return {"ok": True}