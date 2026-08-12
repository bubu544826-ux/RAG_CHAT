const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const loadingMessage = document.querySelector("#loading");
const errorMessage = document.querySelector("#error");
const answerOutput = document.querySelector("#answer");
const sourcesOutput = document.querySelector("#sources");

function setLoading(isLoading) {
  loadingMessage.hidden = !isLoading;
  sendButton.disabled = isLoading;
  questionInput.disabled = isLoading;
}

function renderSources(sources) {
  sourcesOutput.replaceChildren();

  if (sources.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No sources returned.";
    sourcesOutput.append(item);
    return;
  }

  for (const source of sources) {
    const item = document.createElement("li");
    const score = Number(source.score).toFixed(3);
    item.textContent = `${source.source} — ${source.chunk_id} (score: ${score})`;
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  errorMessage.hidden = true;
  setLoading(true);

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
  } catch (error) {
    errorMessage.textContent = error instanceof Error ? error.message : "Something went wrong.";
    errorMessage.hidden = false;
  } finally {
    setLoading(false);
    questionInput.focus();
  }
});
