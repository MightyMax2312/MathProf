import React, { useState, useEffect } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

const App = () => {
  const [question, setQuestion] = useState("");
  const [solution, setSolution] = useState(
    "Ask your Math Professor Agent a question to begin!"
  );
  const [source, setSource] = useState("");
  const [confidence, setConfidence] = useState(0.0);
  const [statusNote, setStatusNote] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setSolution("Analyzing problem...");
    setSource("");
    setConfidence(0.0);
    setStatusNote("");

    try {
      const response = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });
      const data = await response.json();
      setSolution(data.solution);
      setSource(data.source);
      setConfidence(data.confidence);
      setStatusNote(data.status_note);
    } catch (e) {
      setSolution("System Error: Could not connect to agent.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const container = document.getElementById("solution-content");
    if (container && solution) {
      try {
        let cleaned = solution
          .replace(/\\\\/g, "\\") 
          .replace(/^#+\s?/gm, "") 
          .replace(/\*/g, "") 
          .replace(/\n{2,}/g, "<br><br>") 
          .trim();

        const rendered = cleaned.replace(
          /\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g,
          (_, displayMath, inlineMath) => {
            let mathContent = displayMath || inlineMath;

            mathContent = mathContent
              .replace(/<b>|<\/b>/gi, "")
              .replace(/<strong>|<\/strong>/gi, "");

            try {
              return katex.renderToString(mathContent, {
                throwOnError: false,
                displayMode: !!displayMath,
              });
            } catch {
              return mathContent;
            }
          }
        );

        container.innerHTML = rendered;
      } catch {
        container.innerHTML = solution;
      }
    }
  }, [solution]);

  const styles = `
    body {
      background-color: #262624;
      font-family: 'Inter', sans-serif;
      color: #faf9f5;
      margin: 0;
      padding: 0;
    }

    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 4rem 1rem;
    }

    h1 {
      font-size: 3rem;
      font-weight: 800;
      background: linear-gradient(to right, #faf9f5, #faf9f5);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }

    p.subtitle {
      color: #9ca3af;
      font-size: 1rem;
      margin-bottom: 2rem;
      text-align: center;
    }

    .chat-box, .solution-box {
      background-color: #262624;
      border: 1px solid #30363d;
      width: 960px;
      max-width: 96%;
      border-radius: 16px;
      box-shadow: 0 0 18px rgba(0,0,0,0.6);
      margin-bottom: 2rem;
      transition: all 0.3s ease-in-out;
    }

    .chat-box:hover, .solution-box:hover {
      border-color: #d97757;
      box-shadow: 0 0 24px rgba(59,130,246,0.3);
    }

    textarea {
      width: 100%;
      height: 100px;
      background: transparent;
      border: none;
      color: #f3f4f6;
      padding: 1.2rem;
      font-size: 1.1rem;
      resize: none;
      outline: none;
    }

    textarea::placeholder {
      color: #6b7280;
      text-align: center;
    }

    button {
      width: 100%;
      background: linear-gradient(90deg, #d97757, #d97757);
      color: white;
      font-weight: 600;
      font-size: 1.1rem;
      padding: 1rem;
      border: none;
      border-radius: 0 0 16px 16px;
      cursor: pointer;
      transition: 0.3s ease-in-out;
    }

    button:hover {
      background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    }

    .solution-content {
      padding: 2rem;
      text-align: left;
      font-family: "Georgia", serif;
      line-height: 1.8;
      color: #d1d5db;
      overflow-wrap: break-word;
      overflow-x: auto;
      white-space: normal;
      word-spacing: 0.05rem;
    }

    .solution-content strong {
      color: #fbbf24;
      font-weight: 600;
    }

    .katex {
      font-size: 1rem !important;
      white-space: nowrap;
    }

    .katex-display {
      margin: 1rem auto !important;
      text-align: center;
      overflow-x: auto;
      font-size: 1rem !important;
      white-space: nowrap;
    }

    footer {
      width: 960px;
      max-width: 96%;
      color: #9ca3af;
      font-size: 0.9rem;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      border-top: 1px solid #30363d;
      padding-top: 1rem;
    }

    .alert {
      width: 960px;
      max-width: 96%;
      background: rgba(202,138,4,0.1);
      border: 1px solid rgba(234,179,8,0.3);
      color: #facc15;
      border-radius: 12px;
      padding: 1rem;
      margin-top: 1.5rem;
      text-align: center;
      font-size: 0.9rem;
    }

    @media (max-width: 640px) {
      h1 { font-size: 2.2rem; }
      textarea { height: 80px; font-size: 1rem; }
      button { font-size: 1rem; }
    }
  `;

  return (
    <div className="container">
      <style>{styles}</style>

      <h1>Math Professor Agent 🎓</h1>
      <p className="subtitle">
        Agentic RAG with Local LLM + LangGraph + DSPy
      </p>

      <form onSubmit={handleSubmit} className="chat-box">
        <textarea
          placeholder="Enter a math question (e.g., 'Solve 3x + 9 = 21' for KB, or 'Explain the mathematical purpose of the Attention mechanism' for Web Search)"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing Problem..." : "Ask the Professor"}
        </button>
      </form>

      <div className="solution-box">
        <h2
          style={{
            fontSize: "1.5rem",
            fontWeight: 600,
            color: "#93c5fd",
            textAlign: "center",
            paddingTop: "1.5rem",
          }}
        >
          Step-by-Step Solution
        </h2>
        <div id="solution-content" className="solution-content"></div>
      </div>

      <footer>
        <div>
          <strong>Source Path:</strong> {source || "Awaiting Input"}
        </div>
        <div>
          <strong>Confidence:</strong>{" "}
          {confidence ? `${(confidence * 100).toFixed(1)}%` : "--"}
        </div>
        <div>
          <strong>Agent Status:</strong> {statusNote || "Ready"}
        </div>
      </footer>

      <div className="alert">
        <strong>Human-in-the-Loop:</strong> Confidence below 75% triggers DSPy
        self-correction (and flags this for manual review in a production setup).
      </div>
    </div>
  );
};

export default App;
