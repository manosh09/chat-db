import os
import re
import json
from contextlib import asynccontextmanager
from datetime import datetime

import asyncpg
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-3.6-flash"

PORT = int(os.getenv("PORT", 3000))

SCHEMA_TTL_SECONDS = 5 * 60
schema_cache = {"text": None, "fetched_at": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )
    yield
    await app.state.pool.close()


app = FastAPI(title="Chat with your database", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sql: str | None = None
    rowCount: int | None = None
    rows: list[dict] | None = None


async def get_schema_description(pool: asyncpg.Pool) -> str:
    is_fresh = (
        schema_cache["text"] is not None
        and datetime.now().timestamp() - schema_cache["fetched_at"] < SCHEMA_TTL_SECONDS
    )
    if is_fresh:
        return schema_cache["text"]

    rows = await pool.fetch("""
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
        """)

    by_table: dict[str, list[str]] = {}
    for row in rows:
        by_table.setdefault(row["table_name"], []).append(
            f'{row["column_name"]} ({row["data_type"]})'
        )

    text = "\n".join(
        f'Table "{table}": {", ".join(cols)}' for table, cols in by_table.items()
    )

    schema_cache["text"] = text
    schema_cache["fetched_at"] = datetime.now().timestamp()
    return text


def assert_safe_select(sql: str) -> str:
    trimmed = re.sub(r";+\s*$", "", sql.strip())

    if ";" in trimmed:
        raise ValueError("Multiple statements are not allowed.")
    if not re.match(r"^(select|with)\b", trimmed, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    blocked = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create|copy|call|do|vacuum)\b",
        re.IGNORECASE,
    )
    if blocked.search(trimmed):
        raise ValueError(
            "That request would modify data or schema, which is not permitted."
        )
    return trimmed


def extract_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    json_str = match.group(1).strip() if match else text.strip()
    return json.loads(json_str)


async def generate_sql(question: str, schema: str) -> dict:
    prompt = f"""You translate a user's question into a single read-only PostgreSQL query.

Database schema:
{schema}


User's question: {question}"""

    model = genai.GenerativeModel(
        MODEL, generation_config={"response_mime_type": "application/json"}
    )
    response = await run_in_threadpool(model.generate_content, prompt)
    text = response.text or "{}"

    try:
        return extract_json(text)
    except (json.JSONDecodeError, AttributeError):
        raise ValueError(
            "Could not understand how to translate that question into SQL."
        )


async def summarize_results(question: str, sql: str, rows: list[dict]) -> str:
    prompt = f"""You explain database query results to a non-technical user in plain, natural English.
Be concise and direct. Reference actual numbers/names from the data. Do not mention SQL, tables, or columns by name unless the user did.
If there are zero rows, say plainly that nothing matched.

Question: {question}
SQL that was run: {sql}
Result rows (JSON, truncated to first 50): {json.dumps(rows[:50], default=str)}
Total rows returned: {len(rows)}"""

    model = genai.GenerativeModel(MODEL)
    response = await run_in_threadpool(model.generate_content, prompt)
    return (response.text or "").strip()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message
    if not message or not message.strip():
        raise HTTPException(
            status_code=400, detail='Missing "message" in request body.'
        )

    pool: asyncpg.Pool = app.state.pool

    try:
        schema = await get_schema_description(pool)
        parsed = await generate_sql(message, schema)
        sql = parsed.get("sql")
        note = parsed.get("note")

        if not sql:
            return ChatResponse(
                answer=note or "I can't answer that from this database.", sql=None
            )

        safe_sql = assert_safe_select(sql)

        async with pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                await conn.execute("SET LOCAL statement_timeout = '5000'")
                records = await conn.fetch(safe_sql)
                rows = [dict(r) for r in records]

        answer = await summarize_results(message, safe_sql, rows)

        return ChatResponse(
            answer=answer,
            sql=safe_sql,
            rowCount=len(rows),
            rows=[
                {k: str(v) if isinstance(v, datetime) else v for k, v in r.items()}
                for r in rows[:50]
            ],
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err) or "Something went wrong.")


@app.get("/api/health")
async def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory="public", html=True), name="public")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
