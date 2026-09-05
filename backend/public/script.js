const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

// Auto-grow the textarea as the user types
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
});

// Enter sends, Shift+Enter makes a new line
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

function addUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "msg msg--user";
  msg.innerHTML = `<div class="bubble"></div>`;
  msg.querySelector(".bubble").textContent = text;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function addTypingIndicator() {
  const msg = document.createElement("div");
  msg.className = "msg msg--bot typing";
  msg.innerHTML = `<div class="bubble">Thinking…</div>`;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function addBotMessage({ answer, sql, rowCount }, isError = false) {
  const msg = document.createElement("div");
  msg.className = `msg msg--bot${isError ? " msg--error" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = answer;
  msg.appendChild(bubble);

  if (sql) {
    const toggle = document.createElement("div");
    toggle.className = "query-toggle";
    toggle.textContent = "View query used";

    const panel = document.createElement("div");
    panel.className = "query-panel";
    panel.textContent = sql;

    if (typeof rowCount === "number") {
      const rowInfo = document.createElement("span");
      rowInfo.className = "row-count";
      rowInfo.textContent = `${rowCount} row${rowCount === 1 ? "" : "s"} returned`;
      panel.appendChild(rowInfo);
    }

    toggle.addEventListener("click", () => {
      panel.classList.toggle("open");
      toggle.textContent = panel.classList.contains("open")
        ? "Hide query"
        : "View query used";
    });

    msg.appendChild(toggle);
    msg.appendChild(panel);
  }

  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addUserMessage(question);
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  const typingMsg = addTypingIndicator();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    const data = await res.json();

    typingMsg.remove();

    if (!res.ok) {
      addBotMessage({ answer: data.error || "Something went wrong." }, true);
    } else {
      addBotMessage(data);
    }
  } catch (err) {
    typingMsg.remove();
    addBotMessage(
      { answer: "Could not reach the server. Is it running?" },
      true,
    );
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
