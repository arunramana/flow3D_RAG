import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState({});
  const [error, setError] = useState("");
  const nextId = useRef(1);

  async function send(event) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;

    const userId = nextId.current++;
    setMessages((prev) => [...prev, { id: userId, role: "user", text }]);
    setQuestion("");
    setBusy(true);
    setError("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      if (!response.ok) throw new Error("The assistant could not answer.");
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "assistant",
          text: data.answer,
          citations: data.citations,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function toggle(messageId, n) {
    const key = `${messageId}-${n}`;
    setOpen((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  return (
    <div className="app">
      <header>
        <span className="mark">FLOW-3D</span>
        <span>assistant</span>
      </header>

      <main>
        {messages.length === 0 && (
          <p className="empty">Ask about FLOW-3D or CFD. Answers cite the public docs.</p>
        )}
        {messages.map((message) => (
          <article key={message.id} className={message.role}>
            <p>{message.text}</p>
            {message.citations?.length > 0 && (
              <div className="sources">
                {message.citations.map((cite) => {
                  const key = `${message.id}-${cite.n}`;
                  return (
                    <div key={key}>
                      <button type="button" onClick={() => toggle(message.id, cite.n)}>
                        [{cite.n}] {cite.heading}
                      </button>
                      {open[key] && (
                        <div className="passage">
                          <ReactMarkdown>{cite.text}</ReactMarkdown>
                          <a href={cite.url} target="_blank" rel="noreferrer">
                            {cite.url}
                          </a>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </article>
        ))}
        {busy && <p className="waiting">Searching the docs…</p>}
        {error && <p className="error">{error}</p>}
      </main>

      <form onSubmit={send}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="What is the VOF method?"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
