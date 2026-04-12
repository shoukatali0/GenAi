

from flask import Flask, request, jsonify, render_template_string
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv("../.env")

app = Flask(__name__)

# ── Bot personalities ──────────────────────────────────────────────────────────
MODES = {
    "philosopher": {
        "label": "Philosopher",
        "icon": "🧠",
        "system": "You are a wise philosopher bot. You provide deep insights and thoughtful advice on life's big questions.",
        "color": "#6C63FF",
        "accent": "#a89cff",
    },
    "comedian": {
        "label": "Comedian",
        "icon": "😂",
        "system": "You are a comedian bot. You tell jokes and funny stories to make people laugh.",
        "color": "#FF6B35",
        "accent": "#ffaa80",
    },
    "motivator": {
        "label": "Motivator",
        "icon": "🔥",
        "system": "You are a motivational bot. You inspire and encourage people to achieve their goals and overcome challenges.",
        "color": "#11BF7A",
        "accent": "#6DFFBE",
    },
    "advisor": {
        "label": "Financial Advisor",
        "icon": "💰",
        "system": "You are a financial advisor bot. You provide expert advice on personal finance, investments, and money management.",
        "color": "#F5C518",
        "accent": "#ffe57a",
    },
}

# Per-session history stored in memory (keyed by mode; resets on server restart)
chat_histories: dict[str, list] = {k: [] for k in MODES}

# Lazy-load the LLM once
_model = None

def get_model():
    global _model
    if _model is None:
        llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",
            temperature=0.9,
            max_tokens=512,
        )
        _model = ChatHuggingFace(llm=llm)
    return _model


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    mode_key = data.get("mode", "philosopher")
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    mode = MODES.get(mode_key)
    if not mode:
        return jsonify({"error": "Unknown mode"}), 400

    # Build message list: system + history + new user turn
    history = chat_histories[mode_key]
    messages = [SystemMessage(content=mode["system"])] + history + [HumanMessage(content=user_msg)]

    try:
        response = get_model().invoke(messages)
        reply = response.content

        # Persist history
        chat_histories[mode_key].append(HumanMessage(content=user_msg))
        chat_histories[mode_key].append(AIMessage(content=reply))

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    mode_key = request.json.get("mode")
    if mode_key in chat_histories:
        chat_histories[mode_key] = []
    return jsonify({"ok": True})


# ── Embedded HTML/CSS/JS ───────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MindShift · Multi-Personality AI</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:ital,wght@0,400;1,400&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0d0d12;
    --surface: #16161f;
    --border: #252535;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --active: #6C63FF;
    --active-glow: rgba(108,99,255,0.25);
    --radius: 14px;
    --font-ui: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-ui);
    height: 100dvh;
    display: grid;
    grid-template-rows: auto 1fr;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--surface);
  }
  header h1 { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.5px; }
  header span.dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--active); animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.3; }
  }

  /* ── Main layout ── */
  main {
    display: grid;
    grid-template-columns: 220px 1fr;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  aside {
    border-right: 1px solid var(--border);
    padding: 20px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
    background: var(--surface);
  }
  aside p.label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 6px 6px;
  }

  .mode-btn {
    width: 100%;
    padding: 12px 14px;
    border-radius: var(--radius);
    border: 1.5px solid transparent;
    background: transparent;
    color: var(--text);
    font-family: var(--font-ui);
    font-size: 0.9rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    transition: all 0.18s ease;
    text-align: left;
  }
  .mode-btn .icon { font-size: 1.2rem; }
  .mode-btn:hover { background: var(--border); }
  .mode-btn.active {
    border-color: var(--active);
    background: var(--active-glow);
    color: #fff;
    box-shadow: 0 0 16px var(--active-glow);
  }

  aside .reset-btn {
    margin-top: auto;
    padding: 10px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: var(--font-ui);
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  aside .reset-btn:hover { border-color: #ff5555; color: #ff5555; }

  /* ── Chat area ── */
  .chat-wrap {
    display: grid;
    grid-template-rows: 1fr auto;
    overflow: hidden;
  }

  #messages {
    overflow-y: auto;
    padding: 28px 32px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    scroll-behavior: smooth;
  }

  .empty-state {
    margin: auto;
    text-align: center;
    color: var(--muted);
  }
  .empty-state .big-icon { font-size: 3.5rem; margin-bottom: 12px; }
  .empty-state h2 { font-size: 1.4rem; font-weight: 800; margin-bottom: 6px; }
  .empty-state p { font-size: 0.85rem; line-height: 1.6; max-width: 280px; margin: 0 auto; }

  .bubble-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    animation: fadeUp 0.25s ease;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .bubble-row.user { flex-direction: row-reverse; }

  .avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    background: var(--border);
  }
  .bubble-row.user .avatar { background: var(--active); }

  .bubble {
    max-width: 68%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.65;
    font-family: var(--font-mono);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .bubble-row.bot  .bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .bubble-row.user .bubble {
    background: var(--active);
    color: #fff;
    border-bottom-right-radius: 4px;
  }

  .thinking {
    display: flex; gap: 5px; padding: 14px 18px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; border-bottom-left-radius: 4px;
    width: fit-content;
  }
  .thinking span {
    width: 7px; height: 7px; border-radius: 50%; background: var(--muted);
    animation: blink 1.2s infinite;
  }
  .thinking span:nth-child(2) { animation-delay: 0.2s; }
  .thinking span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink {
    0%,80%,100% { opacity: 0.2; } 40% { opacity: 1; }
  }

  /* ── Input bar ── */
  .input-bar {
    padding: 16px 28px 20px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
    background: var(--bg);
    align-items: flex-end;
  }
  #user-input {
    flex: 1;
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 0.9rem;
    padding: 12px 16px;
    resize: none;
    outline: none;
    max-height: 120px;
    min-height: 46px;
    transition: border-color 0.15s;
    line-height: 1.5;
  }
  #user-input:focus { border-color: var(--active); }
  #user-input::placeholder { color: var(--muted); }

  #send-btn {
    height: 46px; width: 46px;
    border-radius: 12px;
    border: none;
    background: var(--active);
    color: #fff;
    font-size: 1.2rem;
    cursor: pointer;
    flex-shrink: 0;
    transition: opacity 0.15s, transform 0.1s;
    display: flex; align-items: center; justify-content: center;
  }
  #send-btn:hover { opacity: 0.85; }
  #send-btn:active { transform: scale(0.94); }
  #send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  /* ── Mobile ── */
  @media (max-width: 640px) {
    main { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
    aside { flex-direction: row; border-right: none; border-bottom: 1px solid var(--border);
            padding: 10px; overflow-x: auto; white-space: nowrap; }
    aside p.label, aside .reset-btn { display: none; }
    .mode-btn { flex-direction: column; gap: 4px; font-size: 0.75rem; padding: 8px 12px; }
    #messages { padding: 16px; }
    .input-bar { padding: 10px 14px 14px; }
  }
</style>
</head>
<body>

<header>
  <span class="dot"></span>
  <h1>MindShift AI</h1>
</header>

<main>
  <!-- Sidebar -->
  <aside>
    <p class="label">Personality</p>
    <button class="mode-btn active" data-mode="philosopher" style="--c:#6C63FF">
      <span class="icon">🧠</span> Philosopher
    </button>
    <button class="mode-btn" data-mode="comedian" style="--c:#FF6B35">
      <span class="icon">😂</span> Comedian
    </button>
    <button class="mode-btn" data-mode="motivator" style="--c:#11BF7A">
      <span class="icon">🔥</span> Motivator
    </button>
    <button class="mode-btn" data-mode="advisor" style="--c:#F5C518">
      <span class="icon">💰</span> Financial Advisor
    </button>
    <button class="reset-btn" onclick="resetChat()">🗑 Clear chat</button>
  </aside>

  <!-- Chat -->
  <div class="chat-wrap">
    <div id="messages">
      <div class="empty-state" id="empty-state">
        <div class="big-icon">🧠</div>
        <h2>Philosopher mode</h2>
        <p>Ask me anything about life, meaning, and existence.</p>
      </div>
    </div>

    <div class="input-bar">
      <textarea id="user-input" rows="1" placeholder="Type a message…"></textarea>
      <button id="send-btn" onclick="sendMessage()">➤</button>
    </div>
  </div>
</main>

<script>
const MODES = {
  philosopher: { label: "Philosopher",       icon: "🧠", hint: "Ask me anything about life, meaning, and existence.",       color: "#6C63FF" },
  comedian:    { label: "Comedian",           icon: "😂", hint: "Tell me something — I'll make it funny.",                   color: "#FF6B35" },
  motivator:   { label: "Motivator",          icon: "🔥", hint: "Share your challenge — let's crush it together.",           color: "#11BF7A" },
  advisor:     { label: "Financial Advisor",  icon: "💰", hint: "Ask me about savings, investing, or budgeting.",            color: "#F5C518" },
};

let currentMode = "philosopher";
const histories = { philosopher: [], comedian: [], motivator: [], advisor: [] };

// ── Mode switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    currentMode = btn.dataset.mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    // Update CSS active color
    const m = MODES[currentMode];
    document.documentElement.style.setProperty("--active", m.color);
    document.documentElement.style.setProperty("--active-glow", m.color + "33");

    renderHistory();
  });
});

// ── Textarea auto-resize + Enter to send ──────────────────────────────────────
const textarea = document.getElementById("user-input");
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
});
textarea.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Render current mode's history ─────────────────────────────────────────────
function renderHistory() {
  const box = document.getElementById("messages");
  const m = MODES[currentMode];
  const hist = histories[currentMode];

  if (!hist.length) {
    box.innerHTML = `
      <div class="empty-state" id="empty-state">
        <div class="big-icon">${m.icon}</div>
        <h2>${m.label} mode</h2>
        <p>${m.hint}</p>
      </div>`;
    return;
  }

  box.innerHTML = "";
  hist.forEach(({ role, text }) => addBubble(role, text, false));
  box.scrollTop = box.scrollHeight;
}

function addBubble(role, text, scroll = true) {
  const box = document.getElementById("messages");
  const es = document.getElementById("empty-state");
  if (es) es.remove();

  const m = MODES[currentMode];
  const row = document.createElement("div");
  row.className = `bubble-row ${role}`;
  row.innerHTML = `
    <div class="avatar">${role === "user" ? "👤" : m.icon}</div>
    <div class="bubble">${escHtml(text)}</div>`;
  box.appendChild(row);
  if (scroll) box.scrollTop = box.scrollHeight;
}

function showThinking() {
  const box = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = "bubble-row bot";
  row.id = "thinking-row";
  row.innerHTML = `<div class="avatar">${MODES[currentMode].icon}</div>
    <div class="thinking"><span></span><span></span><span></span></div>`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function removeThinking() {
  const el = document.getElementById("thinking-row");
  if (el) el.remove();
}

// ── Send message ───────────────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById("user-input");
  const msg = input.value.trim();
  if (!msg) return;

  input.value = "";
  input.style.height = "auto";
  document.getElementById("send-btn").disabled = true;

  histories[currentMode].push({ role: "user", text: msg });
  addBubble("user", msg);
  showThinking();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: currentMode, message: msg }),
    });
    const data = await res.json();
    removeThinking();

    if (data.error) {
      addBubble("bot", "⚠️ Error: " + data.error);
    } else {
      histories[currentMode].push({ role: "bot", text: data.reply });
      addBubble("bot", data.reply);
    }
  } catch (err) {
    removeThinking();
    addBubble("bot", "⚠️ Network error. Is the server running?");
  }

  document.getElementById("send-btn").disabled = false;
  input.focus();
}

// ── Reset ──────────────────────────────────────────────────────────────────────
async function resetChat() {
  await fetch("/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: currentMode }),
  });
  histories[currentMode] = [];
  renderHistory();
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)