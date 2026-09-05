require("dotenv").config();

const express = require("express");
const cors = require("cors");
const { Pool } = require("pg");
const Anthropic = require("@anthropic-ai/sdk");

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static("public"));

// ---------- Postgres ----------
// For safety, connect with a database role that only has SELECT
// privileges. The app also double-checks every query below, but the
// database-level restriction is your real line of defense.
const pool = new Pool({
  host: process.env.PGHOST,
  port: process.env.PGPORT,
  database: process.env.PGDATABASE,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
});

// ---------- Claude ----------
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const MODEL = "claude-sonnet-4-5";

// ---------- Schema introspection ----------
// We feed Claude a description of the actual tables/columns so it writes
// SQL against your real schema instead of guessing. Cached for 5 minutes
// so we're not hitting information_schema on every request.
let schemaCache = { text: null, fetchedAt: 0 };
const SCHEMA_TTL_MS = 5 * 60 * 1000;

async function getSchemaDescription() {
  const isFresh =
    schemaCache.text && Date.now() - schemaCache.fetchedAt < SCHEMA_TTL_MS;
  if (isFresh) return schemaCache.text;

  const { rows } = await pool.query(`
    SELECT table_name, column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position;
  `);

  const byTable = {};
  for (const row of rows) {
    if (!byTable[row.table_name]) byTable[row.table_name] = [];
    byTable[row.table_name].push(`${row.column_name} (${row.data_type})`);
  }

  const text = Object.entries(byTable)
    .map(([table, cols]) => `Table "${table}": ${cols.join(", ")}`)
    .join("\n");

  schemaCache = { text, fetchedAt: Date.now() };
  return text;
}

// ---------- Safety check ----------
// Only ever allow read-only, single-statement queries. This is a
// defense-in-depth check on top of the read-only DB role - never rely
// on this alone.
function assertSafeSelect(sql) {
  const trimmed = sql.trim().replace(/;+\s*$/, "");

  if (trimmed.includes(";")) {
    throw new Error("Multiple statements are not allowed.");
  }
  if (!/^(select|with)\b/i.test(trimmed)) {
    throw new Error("Only SELECT queries are allowed.");
  }
  const blocked =
    /\b(insert|update|delete|drop|alter|truncate|grant|revoke|create|copy|call|do|vacuum)\b/i;
  if (blocked.test(trimmed)) {
    throw new Error(
      "That request would modify data or schema, which is not permitted.",
    );
  }
  return trimmed;
}

// ---------- Step 1: English -> SQL ----------
async function generateSql(question, schema) {
  const system = `You translate a user's question into a single read-only PostgreSQL query.

Database schema:
${schema}

Rules:
- Output ONLY a JSON object, no other text: {"sql": "...", "note": "..."}
- "sql" must be one single SELECT (or WITH ... SELECT) statement. Never write INSERT/UPDATE/DELETE/DDL.
- If the question can't be answered from this schema, set "sql" to null and explain why in "note".
- Add a LIMIT 200 if the query could return many rows and the user didn't ask for a specific count.
- Use only tables/columns that appear in the schema above.`;

  const response = await anthropic.messages.create({
    model: MODEL,
    max_tokens: 500,
    system,
    messages: [{ role: "user", content: question }],
  });

  const text = response.content.find((b) => b.type === "text")?.text || "{}";
  const cleaned = text.replace(/```json|```/g, "").trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (err) {
    throw new Error(
      "Could not understand how to translate that question into SQL.",
    );
  }
  return parsed; // { sql, note }
}

// ---------- Step 2: SQL results -> English ----------
async function summarizeResults(question, sql, rows) {
  const system = `You explain database query results to a non-technical user in plain, natural English.
Be concise and direct. Reference actual numbers/names from the data. Do not mention SQL, tables, or columns by name unless the user did.
If there are zero rows, say plainly that nothing matched.`;

  const userContent = `Question: ${question}
SQL that was run: ${sql}
Result rows (JSON, truncated to first 50): ${JSON.stringify(rows.slice(0, 50))}
Total rows returned: ${rows.length}`;

  const response = await anthropic.messages.create({
    model: MODEL,
    max_tokens: 600,
    system,
    messages: [{ role: "user", content: userContent }],
  });

  return response.content.find((b) => b.type === "text")?.text?.trim() || "";
}

// ---------- API ----------
app.post("/api/chat", async (req, res) => {
  const { message } = req.body;
  if (!message || typeof message !== "string") {
    return res
      .status(400)
      .json({ error: 'Missing "message" in request body.' });
  }

  try {
    const schema = await getSchemaDescription();
    const { sql, note } = await generateSql(message, schema);

    if (!sql) {
      return res.json({
        answer: note || "I can't answer that from this database.",
        sql: null,
      });
    }

    const safeSql = assertSafeSelect(sql);

    // Belt-and-suspenders: run inside an explicitly read-only transaction
    // with a timeout, even though the role and the regex check above
    // should already prevent writes and runaway queries.
    const client = await pool.connect();
    let rows;
    try {
      await client.query("BEGIN TRANSACTION READ ONLY");
      await client.query("SET LOCAL statement_timeout = '5s'");
      const result = await client.query(safeSql);
      rows = result.rows;
      await client.query("COMMIT");
    } catch (err) {
      await client.query("ROLLBACK");
      throw err;
    } finally {
      client.release();
    }

    const answer = await summarizeResults(message, safeSql, rows);

    res.json({
      answer,
      sql: safeSql,
      rowCount: rows.length,
      rows: rows.slice(0, 50),
    });
  } catch (err) {
    console.error(err);
    res.status(400).json({ error: err.message || "Something went wrong." });
  }
});

app.get("/api/health", (_req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log(`Chat-with-database running on http://localhost:${PORT}`),
);
