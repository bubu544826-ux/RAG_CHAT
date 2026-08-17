const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const clearQuestionButton = document.querySelector("#clear-question");
const loadingMessage = document.querySelector("#loading");
const errorMessage = document.querySelector("#error");
const answerOutput = document.querySelector("#answer");
const answerSkeleton = document.querySelector("#answer-skeleton");
const copyButton = document.querySelector("#copy-answer");
const copyLabel = copyButton.querySelector(".copy-label");
const latencyOutput = document.querySelector("#latency");
const sourcesOutput = document.querySelector("#sources");
const sourceCount = document.querySelector("#source-count");
const themeToggle = document.querySelector("#theme-toggle");
const exampleChips = document.querySelector("#examples");
const historyPanel = document.querySelector("#history-panel");
const historyList = document.querySelector("#history");
const clearHistoryButton = document.querySelector("#clear-history");

const THEME_KEY = "rag-theme";
const HISTORY_KEY = "rag-history";
const HISTORY_LIMIT = 8;

let history = [];
let copyResetTimer = 0;

/* ---------- theme ---------- */

function applyTheme(mode) {
  if (mode === "dark" || mode === "light") {
    document.documentElement.dataset.theme = mode;
  } else {
    delete document.documentElement.dataset.theme;
  }

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const isDark = mode === "dark" || (mode !== "light" && prefersDark);
  themeToggle.setAttribute("aria-pressed", String(isDark));
}

function storedTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch {
    return null;
  }
}

function initTheme() {
  applyTheme(storedTheme());

  themeToggle.addEventListener("click", () => {
    const isDark = themeToggle.getAttribute("aria-pressed") === "true";
    const next = isDark ? "light" : "dark";

    applyTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* storage unavailable; the choice simply will not persist */
    }
  });
}

/* ---------- history ---------- */

function readHistory() {
  try {
    const raw = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((item) => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function renderHistory() {
  historyList.replaceChildren();
  historyPanel.hidden = history.length === 0;

  for (const question of history) {
    const item = document.createElement("li");
    const button = document.createElement("button");

    button.type = "button";
    button.textContent = question;
    button.title = question;
    button.addEventListener("click", () => {
      questionInput.value = question;
      syncClearButton();
      askQuestion(question);
    });

    item.append(button);
    historyList.append(item);
  }
}

function rememberQuestion(question) {
  history = [question, ...history.filter((item) => item !== question)].slice(0, HISTORY_LIMIT);

  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    /* storage unavailable; history stays in memory only */
  }

  renderHistory();
}

/* ---------- view state ---------- */

function syncClearButton() {
  clearQuestionButton.hidden = questionInput.value.length === 0;
}

function setLoading(isLoading) {
  loadingMessage.hidden = !isLoading;
  sendButton.disabled = isLoading;
  questionInput.disabled = isLoading;
  answerSkeleton.hidden = !isLoading;
  answerOutput.hidden = isLoading;

  for (const chip of exampleChips.querySelectorAll(".chip")) {
    chip.disabled = isLoading;
  }

  if (isLoading) {
    copyButton.hidden = true;
    latencyOutput.textContent = "";
  }
}

function renderSources(sources) {
  sourcesOutput.replaceChildren();
  sourceCount.textContent = String(sources.length);

  if (sources.length === 0) {
    const item = document.createElement("li");
    item.className = "source-empty";
    item.textContent = "No sources returned.";
    sourcesOutput.append(item);
    return;
  }

  for (const source of sources) {
    const item = document.createElement("li");
    const row = document.createElement("div");
    const details = document.createElement("div");
    const sourceName = document.createElement("strong");
    const chunkId = document.createElement("span");
    const scoreBadge = document.createElement("span");
    const body = document.createElement("div");
    const bar = document.createElement("div");
    const barFill = document.createElement("span");
    const score = Number(source.score).toFixed(3);
    const ratio = Math.max(0, Math.min(1, Number(source.score) || 0));

    row.className = "source-row";
    details.className = "source-details";
    sourceName.textContent = source.source;
    chunkId.textContent = source.chunk_id;
    scoreBadge.className = "source-score";
    scoreBadge.textContent = score;

    body.className = "source-body";
    body.textContent = `Chunk ${source.chunk_id} · relevance ${score}`;
    body.hidden = true;

    bar.className = "source-bar";
    barFill.style.width = `${ratio * 100}%`;
    bar.append(barFill);

    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.setAttribute("aria-expanded", "false");
    item.addEventListener("click", () => {
      body.hidden = !body.hidden;
      item.classList.toggle("is-open", !body.hidden);
      item.setAttribute("aria-expanded", String(!body.hidden));
    });
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        item.click();
      }
    });

    details.append(sourceName, chunkId);
    row.append(details, scoreBadge);
    item.append(row, body, bar);
    sourcesOutput.append(item);
  }
}

async function readError(response) {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

/* ---------- ask ---------- */

async function askQuestion(question) {
  if (!question || sendButton.disabled) {
    return;
  }

  errorMessage.hidden = true;
  setLoading(true);

  const startedAt = performance.now();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const result = await response.json();
    answerOutput.textContent = result.answer;
    renderSources(result.sources);
    latencyOutput.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
    copyButton.hidden = !navigator.clipboard;
    rememberQuestion(question);
  } catch (error) {
    errorMessage.textContent = error instanceof Error ? error.message : "Something went wrong.";
    errorMessage.hidden = false;
  } finally {
    setLoading(false);
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(questionInput.value.trim());
});

exampleChips.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip) {
    return;
  }

  questionInput.value = chip.textContent.trim();
  syncClearButton();
  askQuestion(questionInput.value);
});

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(answerOutput.textContent);
    copyLabel.textContent = "Copied";
    copyButton.classList.add("is-copied");

    clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(() => {
      copyLabel.textContent = "Copy";
      copyButton.classList.remove("is-copied");
    }, 1600);
  } catch {
    copyLabel.textContent = "Press Ctrl+C";
  }
});

clearHistoryButton.addEventListener("click", () => {
  history = [];
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    /* nothing to clean up when storage is unavailable */
  }
  renderHistory();
  questionInput.focus();
});

clearQuestionButton.addEventListener("click", () => {
  questionInput.value = "";
  syncClearButton();
  questionInput.focus();
});

questionInput.addEventListener("input", syncClearButton);

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && questionInput.value) {
    event.preventDefault();
    questionInput.value = "";
    syncClearButton();
  }
});

document.addEventListener("keydown", (event) => {
  const target = event.target;
  const isTyping = target instanceof HTMLElement &&
    (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);

  if (event.key === "/" && !isTyping && !event.metaKey && !event.ctrlKey) {
    event.preventDefault();
    questionInput.focus();
  }
});

initTheme();
history = readHistory();
renderHistory();
syncClearButton();
