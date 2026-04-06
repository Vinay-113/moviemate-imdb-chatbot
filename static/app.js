const starterPrompts = [
  "Suggest sci-fi movies released after 2010",
  "Movies similar to Inception",
  "Good drama movies with high IMDb ratings",
  "Movies starring Leonardo DiCaprio",
  "Show me thriller movies under 120 minutes",
  "Who directed The Godfather?",
];

let sessionId = null;

const summaryCards = document.getElementById("summary-cards");
const notesContainer = document.getElementById("data-notes");
const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const filters = document.getElementById("active-filters");
const starterPromptContainer = document.getElementById("starter-prompts");
const newChatButton = document.getElementById("new-chat-button");
const template = document.getElementById("message-template");

const numberFormat = new Intl.NumberFormat("en-US");

function addStarterPrompts() {
  starterPrompts.forEach((prompt) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = prompt;
    button.addEventListener("click", () => {
      input.value = prompt;
      input.focus();
    });
    starterPromptContainer.appendChild(button);
  });
}

function addMessage(role, text, cards = []) {
  const fragment = template.content.cloneNode(true);
  const root = fragment.querySelector(".message");
  root.classList.add(role);
  fragment.querySelector(".message-role").textContent = role === "assistant" ? "MovieMate" : "You";
  fragment.querySelector(".message-body").textContent = text;

  const cardGrid = fragment.querySelector(".card-grid");
  if (cards.length) {
    cards.forEach((card) => cardGrid.appendChild(buildMovieCard(card)));
  }

  messages.appendChild(fragment);
  messages.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "end" });
}

function buildMovieCard(card) {
  const article = document.createElement("article");
  article.className = "movie-card";

  const genres = card.genres.join(" · ");
  const runtime = card.runtime_minutes ? `${card.runtime_minutes} min` : "Runtime N/A";
  const metascore = card.metascore ?? "N/A";
  const gross = card.gross ? `$${numberFormat.format(card.gross)}` : "Gross N/A";

  article.innerHTML = `
    <div class="card-topline">
      <div class="card-title">${card.title}</div>
      <div class="badge">IMDb ${card.rating}</div>
    </div>
    <div class="badge-row">
      <span class="badge">${card.year ?? "Unknown year"}</span>
      <span class="badge">${genres}</span>
      <span class="badge">${runtime}</span>
      <span class="badge">Metascore ${metascore}</span>
    </div>
    <div class="card-meta">Directed by ${card.director}. Stars ${card.stars.join(", ")}.</div>
    <div class="card-meta">Certificate ${card.certificate} · Votes ${numberFormat.format(card.votes)} · ${gross}</div>
    <div class="card-overview">${card.overview}</div>
  `;
  return article;
}

function renderSummary(summary) {
  summaryCards.innerHTML = "";
  const cards = [
    ["Movies", numberFormat.format(summary.movie_count)],
    ["Avg IMDb Rating", summary.average_rating.toFixed(2)],
    ["Avg Runtime", `${summary.average_runtime.toFixed(1)} min`],
    ["Avg Metascore", summary.average_metascore.toFixed(1)],
  ];

  cards.forEach(([label, value], index) => {
    const card = document.createElement("article");
    card.className = "summary-card";
    card.style.animationDelay = `${index * 80}ms`;
    card.innerHTML = `
      <p class="summary-label">${label}</p>
      <div class="summary-value">${value}</div>
    `;
    summaryCards.appendChild(card);
  });
}

function renderNotes(notes) {
  notesContainer.innerHTML = "";
  notes.forEach((note) => {
    const item = document.createElement("div");
    item.className = "note";
    item.textContent = note;
    notesContainer.appendChild(item);
  });
}

function renderBars(containerId, data) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const maxValue = Math.max(...data.map((entry) => entry.value), 1);
  data.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const width = `${(entry.value / maxValue) * 100}%`;
    row.innerHTML = `
      <div class="bar-header">
        <span class="bar-label">${entry.label}</span>
        <span class="bar-meta">${numberFormat.format(entry.value)}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${width};"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderFilters(filterState) {
  filters.innerHTML = "";
  const pills = [];

  if (filterState.genres.length) pills.push(`Genres: ${filterState.genres.join(", ")}`);
  if (filterState.people.length) pills.push(`People: ${filterState.people.join(", ")}`);
  if (filterState.directors.length) pills.push(`Directors: ${filterState.directors.join(", ")}`);
  if (filterState.year_min !== null) pills.push(`Year >= ${filterState.year_min}`);
  if (filterState.year_max !== null) pills.push(`Year <= ${filterState.year_max}`);
  if (filterState.runtime_min !== null) pills.push(`Runtime >= ${filterState.runtime_min}`);
  if (filterState.runtime_max !== null) pills.push(`Runtime <= ${filterState.runtime_max}`);
  if (filterState.rating_min !== null) pills.push(`IMDb >= ${filterState.rating_min}`);
  if (filterState.rating_max !== null) pills.push(`IMDb <= ${filterState.rating_max}`);
  if (filterState.sort_by && filterState.sort_by !== "relevance") pills.push(`Sort: ${filterState.sort_by}`);

  pills.forEach((text) => {
    const pill = document.createElement("span");
    pill.className = "filter-pill";
    pill.textContent = text;
    filters.appendChild(pill);
  });
}

async function loadInsights() {
  const response = await fetch("/api/insights");
  const payload = await response.json();
  renderSummary(payload.summary);
  renderNotes(payload.data_notes);
  renderBars("top-genres", payload.top_genres);
  renderBars("rating-distribution", payload.rating_distribution);
  renderBars("decades", payload.decades);
  renderBars("top-directors", payload.top_directors);
  renderBars("top-stars", payload.top_stars);
  renderBars("missing-fields", payload.missing_fields);
}

async function sendMessage(message) {
  addMessage("user", message);

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  const payload = await response.json();
  if (payload.error) {
    addMessage("assistant", payload.error);
    return;
  }

  sessionId = payload.session_id;
  renderFilters(payload.filters);
  addMessage("assistant", payload.reply, payload.cards || []);
}

function resetConversation() {
  sessionId = null;
  messages.innerHTML = "";
  filters.innerHTML = "";
  addMessage(
    "assistant",
    "Ask for genres, actors, directors, release years, runtimes, ratings, or movies similar to a title already in the dataset."
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  await sendMessage(message);
});

newChatButton.addEventListener("click", () => {
  resetConversation();
  input.focus();
});

addStarterPrompts();
resetConversation();
loadInsights().catch((error) => {
  addMessage("assistant", `I could not load the dataset insights: ${error.message}`);
});
