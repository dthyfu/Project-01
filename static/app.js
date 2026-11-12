const historyList = document.getElementById("historyList");
const messagesElement = document.getElementById("messages");
const chatScroll = document.getElementById("chatScroll");
const welcome = document.getElementById("welcome");
const questionInput = document.getElementById("question");
const chatForm = document.getElementById("chatForm");
const riskForm = document.getElementById("riskForm");
const riskResult = document.getElementById("riskResult");
const sendButton = document.getElementById("sendButton");
const activity = document.getElementById("activity");
const sidebar = document.getElementById("sidebar");
const scrim = document.getElementById("sidebarScrim");

let currentConversationId = null;
let busy = false;
const activeEvents = new Map();
const agentLabels = {
    supervisor: "Supervisor",
    retrieval: "HybridRAG Retrieval",
    medical: "Medical Research Agent",
    nutrition: "Nutrition Agent",
    data_analysis: "Data Analysis Agent",
    knowledge: "Knowledge Agent",
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[char]));
}

function inlineMarkdown(value) {
    let safe = escapeHtml(value);
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    return safe;
}

// Small allowlisted Markdown renderer: source HTML is escaped before markup is added.
function renderMarkdown(markdown) {
    const lines = String(markdown || "").replace(/\r/g, "").split("\n");
    const blocks = [];
    let paragraph = [];
    let listType = null;
    let listItems = [];
    let codeLines = null;
    const flushParagraph = () => {
        if (paragraph.length) blocks.push(`<p>${paragraph.map(inlineMarkdown).join("<br>")}</p>`);
        paragraph = [];
    };
    const flushList = () => {
        if (listItems.length) blocks.push(`<${listType}>${listItems.map(item => `<li>${inlineMarkdown(item)}</li>`).join("")}</${listType}>`);
        listType = null;
        listItems = [];
    };

    for (let index = 0; index < lines.length; index++) {
        const line = lines[index];
        if (codeLines !== null) {
            if (line.startsWith("```")) {
                blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
                codeLines = null;
            } else codeLines.push(line);
            continue;
        }
        if (line.startsWith("```")) {
            flushParagraph(); flushList(); codeLines = [];
            continue;
        }
        if (line.includes("|") && lines[index + 1]?.match(/^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/)) {
            flushParagraph(); flushList();
            const cells = value => value.trim().replace(/^\||\|$/g, "").split("|").map(cell => cell.trim());
            const headers = cells(line);
            index++;
            const rows = [];
            while (index + 1 < lines.length && lines[index + 1].includes("|")) {
                index++;
                rows.push(cells(lines[index]));
            }
            const head = headers.map(cell => `<th>${inlineMarkdown(cell)}</th>`).join("");
            const body = rows.map(row => `<tr>${headers.map((_, cellIndex) => `<td>${inlineMarkdown(row[cellIndex] || "")}</td>`).join("")}</tr>`).join("");
            blocks.push(`<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
            continue;
        }
        const listMatch = line.match(/^\s*(?:[-*+]\s+|\d+\.\s+)(.*)$/);
        if (listMatch) {
            flushParagraph();
            const type = /^\s*\d+\./.test(line) ? "ol" : "ul";
            if (listType && listType !== type) flushList();
            listType = type;
            listItems.push(listMatch[1]);
            continue;
        }
        flushList();
        if (!line.trim()) { flushParagraph(); continue; }
        const heading = line.match(/^(#{1,3})\s+(.*)$/);
        if (heading) {
            flushParagraph();
            const level = heading[1].length;
            blocks.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        } else paragraph.push(line);
    }
    if (codeLines !== null) blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    flushParagraph(); flushList();
    return blocks.join("");
}

function scrollToBottom() {
    chatScroll.scrollTop = chatScroll.scrollHeight;
}

function makeMessage(role, content, agents = [], status = "completed") {
    const article = document.createElement("article");
    article.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
    article.dataset.status = status;
    const avatar = document.createElement("div");
    avatar.className = `message-avatar ${role === "user" ? "user-avatar" : "assistant-avatar"}`;
    avatar.textContent = role === "user" ? "B" : "D";
    const body = document.createElement("div");
    body.className = "message-body";
    const contentElement = document.createElement("div");
    contentElement.className = "message-content";
    contentElement.innerHTML = renderMarkdown(content);
    body.appendChild(contentElement);
    const notice = statusNotice(status);
    if (role === "assistant" && notice) {
        const state = document.createElement("div");
        state.className = `message-status ${status}`;
        state.textContent = notice;
        body.appendChild(state);
    }
    if (agents.length) {
        const used = document.createElement("div");
        used.className = "agents-used";
        used.textContent = `Đã xử lý: ${agents.map(name => agentLabels[name] || name).join(" · ")}`;
        body.appendChild(used);
    }
    article.append(avatar, body);
    messagesElement.appendChild(article);
    welcome.classList.add("hidden");
    scrollToBottom();
    return { article, contentElement, body };
}

function statusNotice(status) {
    return ({
        fallback: "Câu trả lời được tổng hợp dự phòng từ nội dung đã truy xuất.",
        partial: "Một phần xử lý chưa hoàn tất.",
        insufficient: "Tài liệu hiện có chưa đủ cơ sở để kết luận.",
        error: "Có lỗi khi xử lý câu hỏi.",
    })[status] || "";
}

function updateActivity() {
    activity.replaceChildren();
    if (!activeEvents.size) {
        activity.classList.add("hidden");
        return;
    }
    activity.classList.remove("hidden");
    for (const [name, item] of activeEvents) {
        const row = document.createElement("div");
        row.className = `activity-item ${item.status}`;
        const dot = document.createElement("span");
        dot.className = "activity-dot";
        const label = document.createElement("span");
        label.textContent = item.message || `${agentLabels[name] || name} đang xử lý`;
        row.append(dot, label);
        activity.appendChild(row);
    }
}

function handleEvent(name, payload, assistantBubble) {
    if (name === "status") {
        activeEvents.set(payload.agent, payload);
        updateActivity();
    } else if (name === "final") {
        assistantBubble.contentElement.innerHTML = renderMarkdown(payload.answer || "Chưa có câu trả lời.");
        assistantBubble.article.dataset.status = payload.status || "completed";
        const oldNotice = assistantBubble.body.querySelector(".message-status");
        if (oldNotice) oldNotice.remove();
        const noticeText = statusNotice(payload.status);
        if (noticeText) {
            const notice = document.createElement("div");
            notice.className = `message-status ${payload.status}`;
            notice.textContent = noticeText;
            assistantBubble.body.appendChild(notice);
        }
        const used = document.createElement("div");
        used.className = "agents-used";
        used.textContent = payload.agents_used?.length
            ? `Đã xử lý: ${payload.agents_used.map(agent => agentLabels[agent] || agent).join(" · ")}`
            : (payload.status === "fallback" ? "Tổng hợp từ tài liệu truy xuất" : "Trả lời theo phạm vi hỗ trợ");
        assistantBubble.body.appendChild(used);
        refreshHistory();
    } else if (name === "error") {
        assistantBubble.contentElement.textContent = payload.message || "Đã xảy ra lỗi.";
        assistantBubble.article.dataset.status = "error";
    }
    scrollToBottom();
}

async function streamChat(question) {
    busy = true;
    sendButton.disabled = true;
    questionInput.disabled = true;
    activeEvents.clear();
    updateActivity();
    try {
        makeMessage("user", question);
        const assistantBubble = makeMessage("assistant", "", [], "pending");
        const thinking = document.createElement("p");
        thinking.className = "thinking-indicator";
        thinking.append(document.createTextNode("Đang suy nghĩ"));
        for (let index = 0; index < 3; index++) {
            const dot = document.createElement("span");
            dot.className = "thinking-dot";
            dot.setAttribute("aria-hidden", "true");
            thinking.appendChild(dot);
        }
        assistantBubble.contentElement.appendChild(thinking);
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: currentConversationId, question }),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || "Không thể gửi câu hỏi.");
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let boundary;
            while ((boundary = buffer.indexOf("\n\n")) >= 0) {
                const rawEvent = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);
                let eventName = "message";
                let dataText = "";
                for (const line of rawEvent.split(/\r?\n/)) {
                    if (line.startsWith("event:")) eventName = line.slice(6).trim();
                    if (line.startsWith("data:")) dataText += line.slice(5).trim();
                }
                if (dataText) {
                    try { handleEvent(eventName, JSON.parse(dataText), assistantBubble); }
                    catch { /* Backend keeps the saved response if an event frame is malformed. */ }
                }
            }
        }
        if (assistantBubble.article.dataset.status === "pending") {
            assistantBubble.contentElement.textContent = "Kết nối bị gián đoạn. Hãy mở lại cuộc trò chuyện để tải nội dung đã lưu.";
            assistantBubble.article.dataset.status = "error";
        }
    } catch (error) {
        if (!messagesElement.querySelector('[data-status="pending"]')) makeMessage("assistant", error.message || "Đã xảy ra lỗi.", [], "error");
        else {
            const bubble = messagesElement.querySelector('[data-status="pending"] .message-content');
            if (bubble) bubble.textContent = error.message || "Đã xảy ra lỗi.";
            messagesElement.querySelector('[data-status="pending"]').dataset.status = "error";
        }
    } finally {
        busy = false;
        sendButton.disabled = false;
        questionInput.disabled = false;
        questionInput.focus();
        activeEvents.clear();
        updateActivity();
        refreshHistory();
    }
}

async function createConversation() {
    const response = await fetch("/api/conversations", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "Cuộc trò chuyện mới" }),
    });
    if (!response.ok) throw new Error("Không thể tạo cuộc trò chuyện mới.");
    const data = await response.json();
    currentConversationId = data.conversation.id;
    messagesElement.replaceChildren();
    welcome.classList.remove("hidden");
    closeSidebar();
    refreshHistory();
}

async function openConversation(id) {
    const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`);
    if (!response.ok) return;
    const data = await response.json();
    currentConversationId = id;
    messagesElement.replaceChildren();
    for (const message of data.messages) makeMessage(message.role, message.content, message.agents_used || [], message.status);
    welcome.classList.toggle("hidden", data.messages.length > 0);
    await refreshHistory();
    closeSidebar();
}

async function deleteConversation(id, event) {
    event.stopPropagation();
    if (!window.confirm("Xóa cuộc trò chuyện này?")) return;
    const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!response.ok) return;
    if (currentConversationId === id) {
        currentConversationId = null;
        messagesElement.replaceChildren();
        welcome.classList.remove("hidden");
    }
    refreshHistory();
}

async function refreshHistory() {
    try {
        const response = await fetch("/api/conversations");
        const data = await response.json();
        historyList.replaceChildren();
        for (const conversation of data.conversations || []) {
            const row = document.createElement("div");
            row.className = `history-row ${conversation.id === currentConversationId ? "selected" : ""}`;
            const open = document.createElement("button");
            open.className = "history-open";
            open.textContent = conversation.title || "Cuộc trò chuyện mới";
            open.title = open.textContent;
            open.addEventListener("click", () => openConversation(conversation.id));
            const remove = document.createElement("button");
            remove.className = "history-delete";
            remove.type = "button";
            remove.textContent = "···";
            remove.setAttribute("aria-label", "Xóa cuộc trò chuyện");
            remove.addEventListener("click", event => deleteConversation(conversation.id, event));
            row.append(open, remove);
            historyList.appendChild(row);
        }
    } catch { /* History remains usable when the backend is restarting. */ }
}

function closeSidebar() {
    sidebar.classList.remove("open");
    scrim.classList.remove("visible");
}

chatForm.addEventListener("submit", event => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question || busy) return;
    questionInput.value = "";
    questionInput.style.height = "auto";
    busy = true;
    sendButton.disabled = true;
    questionInput.disabled = true;
    const start = () => streamChat(question);
    if (!currentConversationId) createConversation().then(start).catch(error => {
        busy = false; sendButton.disabled = false; questionInput.disabled = false;
        alert(error.message);
    });
    else start();
});

questionInput.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});
questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 180)}px`;
});
riskForm.addEventListener("submit", async event => {
    event.preventDefault();
    const button = riskForm.querySelector("button[type='submit']");
    const values = Object.fromEntries(new FormData(riskForm).entries());
    for (const field of ["age", "bmi", "HbA1c_level", "blood_glucose_level"]) values[field] = Number(values[field]);
    button.disabled = true;
    button.textContent = "Đang tính…";
    riskResult.classList.remove("hidden", "error");
    riskResult.textContent = "Đang đánh giá dữ liệu đầu vào…";
    try {
        const response = await fetch("/api/predict-risk", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || "Không thể tính ước tính lúc này.");
        riskResult.replaceChildren();
        const score = document.createElement("strong");
        score.textContent = payload.risk_label || "Chưa thể phân loại";
        const note = document.createElement("span");
        note.textContent = payload.message;
        const clinicalNote = document.createElement("p");
        clinicalNote.textContent = payload.clinical_note || "";
        riskResult.append(score, note);
        if (payload.clinical_note) {
            riskResult.classList.add("clinical-warning");
            riskResult.appendChild(clinicalNote);
        } else riskResult.classList.remove("clinical-warning");
    } catch (error) {
        riskResult.classList.add("error");
        riskResult.textContent = error.message || "Không thể tính ước tính lúc này.";
    } finally {
        button.disabled = false;
        button.textContent = "Tính ước tính";
    }
});
document.getElementById("newChat").addEventListener("click", () => createConversation().catch(error => alert(error.message)));
document.querySelectorAll(".suggestion").forEach(button => button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    chatForm.requestSubmit();
}));
document.getElementById("openSidebar").addEventListener("click", () => {
    sidebar.classList.add("open"); scrim.classList.add("visible");
});
document.getElementById("closeSidebar").addEventListener("click", closeSidebar);
scrim.addEventListener("click", closeSidebar);

async function checkHealth() {
    const dot = document.getElementById("healthDot");
    const label = document.getElementById("healthText");
    try {
        const response = await fetch("/api/health");
        const health = await response.json();
        const available = health.ollama === "available";
        dot.classList.toggle("offline", !available);
        const labels = {
            generating: "Đang tạo câu trả lời",
            loading: "Đang nạp model",
            queued: "Đang chờ lượt xử lý",
            loaded: "Sẵn sàng · model đã nạp",
            unloaded: "Sẵn sàng · model sẽ nạp khi cần",
            not_installed: "Không tìm thấy model",
            server_unavailable: "Chưa khả dụng",
        };
        label.textContent = labels[health.model_state]
            || (available ? "Sẵn sàng" : "Chưa khả dụng");
    } catch {
        dot.classList.add("offline"); label.textContent = "Backend chưa khả dụng";
    }
}

refreshHistory();
checkHealth();
window.setInterval(checkHealth, 15000);
