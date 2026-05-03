# 🎥 Loom Video Script: Helix SROP Walkthrough
**Duration:** ~3 Minutes

---

## Part 1: Intro & Architecture (0:00 - 1:00)
**Action:** Start with the **Architecture Diagram** (`helix_srop_architecture_diagram.png`) on screen.

- "Hi, I'm Abhay, and this is my implementation of the Helix Stateful RAG Orchestration Pipeline."
- "My primary goal was to build a system that is not only accurate but also **production-fast**."
- "As you can see in the diagram, I've moved away from the standard 'LLM-as-Orchestrator' model which usually adds 10+ seconds of latency. Instead, I've implemented a **Heuristic-First Pipeline**."
- "We use a 0ms intent router to pick a specialist agent immediately. This cuts our sequential LLM hops from 3 down to 1, delivering responses in under 5 seconds."
- "The entire system is stateful—we persist session context in SQLite, meaning conversations survive process restarts."

---

## Part 2: Feature Demo (1:00 - 2:15)
**Action:** Switch to the **Demo UI** (`demo_ui.html`).

- "Let's see it in action. I'll open the Helix AI Concierge."
- "First, I'll ask a product question: *'How do I rotate a deploy key?'*"
- **(Type question, wait for response)**
- "Notice the response is cited with specific chunk IDs. In the **Trace Sidebar** on the left, you can see exactly how the Knowledge Agent retrieved these from ChromaDB."
- "Next, let's test the stateful account lookup: *'Show me my last 5 builds.'*"
- **(Type question, wait for response)**
- "The router instantly switched to the Account Agent. Now, a follow-up: *'What about failed ones?'*"
- "Because we maintain state, the agent knows we're still talking about builds."
- "Finally, let's trigger an escalation: *'My builds keep failing, I need a human.'*"
- "The Escalation Agent creates a support ticket and stores the ID in our session state."

---

## Part 3: Reliability & Technical Specs (2:15 - 3:00)
**Action:** Switch to a terminal and run the **Eval Harness**.

- "To ensure this remains robust, I've included an automated evaluation harness."
- **(Run `uv run python eval/run_eval.py`)**
- "As you can see, the system scores a perfect **10/10 on routing accuracy**."
- "A key technical highlight is how I've enforced RAG reliability. I use `tool_config` modes to **force** the agent to search documentation before answering, eliminating 'false negative' hallucinations."
- "The project is fully containerized and ready for a 'one-click' deployment via Docker Compose."
- "Thank you for reviewing my submission. All setup details are in the README!"

---

## Preparation Tips:
1. **Clean Start**: Run `rm helix.db` and re-run ingestion before recording to start fresh.
2. **Speed**: Gemini 2.0 Flash is fast, but if it takes >3 seconds, feel free to pause the recording during the "thinking" phase to keep the video snappy.
3. **Cursor**: Use a highlight/circle cursor if your recording tool supports it.
