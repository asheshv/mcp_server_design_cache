# 🤖 AI Agent Usage Examples

This document demonstrates how an AI agent (like Claude Code, Cursor, or Windsurf) uses the **Design Cache MCP Server** during a standard development workflow to save context and retrieve past decisions.

## 1. Starting a New Session
When an AI agent starts a new session for an ongoing project, it can instantly retrieve recent context without needing you to copy-paste the history.

**Agent Action:**
```json
{
  "tool": "get_recent_activity",
  "project": "frontend_migration",
  "limit": 3
}
```

**Agent Thought Process:**
> "I see the last three design decisions were about migrating the auth layer to JWT, moving from Redux to Zustand, and establishing a new components folder structure. I'm now up to speed on the recent `frontend_migration` work."

---

## 2. Storing a New Design Idea (Caching Context)
During a brainstorming session, you and the AI discuss the implementation of a new notification system. Once a decision is reached, the AI caches it for long-term memory.

**Agent Action:**
```json
{
  "tool": "store_note",
  "project": "frontend_migration",
  "title": "Use WebSockets for Real-time Notifications",
  "content": "Decided to use Socket.io instead of Server-Sent Events (SSE) because we need bidirectional communication for read-receipts. The fallback will be standard polling every 30s.",
  "cache_type": "idea"
}
```

---

## 3. Retrieving Past Context via Hybrid Search
Three weeks later, in a completely different chat session, you ask the AI to implement the read-receipts for notifications. The AI searches its memory to recall the exact architectural decision.

**Agent Action:**
```json
{
  "tool": "search_design",
  "project": "frontend_migration",
  "query": "notification read receipts bidirectional communication"
}
```
*The server returns the top matches (abstracts).*

**Agent Follow-up Action:**
```json
{
  "tool": "expand_design_note",
  "cache_id": "018e4b7a-9c40-7e00-a15b-5582e2fa9b61"
}
```

**Agent Thought Process:**
> "Ah, I found the exact note from 3 weeks ago. We decided to use WebSockets natively for read-receipts with a 30s polling fallback. I will implement it exactly this way."

---

## 4. Graduating an Idea to a Formal Spec/ADR
Once enough ideas have gathered around a topic, you ask the AI to write a formal Technical Specification or Architecture Decision Record (ADR) based on the chat history.

**Agent Action:**
```json
{
  "tool": "generate_adr_from_cache",
  "idea_id": "018e4b7a-9c40-7e00-a15b-5582e2fa9b61",
  "status": "accepted"
}
```
*The server returns formatted Markdown. The AI saves this to `docs/ADRs/003-websockets.md`.*

**Agent Follow-up Action:**
```json
{
  "tool": "sync_doc_status",
  "cache_id": "018e4b7a-9c40-7e00-a15b-5582e2fa9b61",
  "file_path": "/absolute/path/to/docs/ADRs/003-websockets.md",
  "status": "implemented"
}
```

**Agent Thought Process:**
> "I have successfully drafted the ADR to the filesystem and updated the database cache to link back and point to the official `003-websockets.md` file. The idea is now formalized."

---

## 5. Cleaning Up Old Ideas (Context Compression)
To keep the database fast and the LLM's context window extremely focused, the AI can summarize dozens of scattered brainstorming notes into a single "Project Level" note, deleting the granular history.

**Agent Action:**
```json
{
  "tool": "summarize_and_cleanup",
  "project": "frontend_migration",
  "ids_to_summarize": [
    "018e4b7a-9c40-7e00-a15b-5582...", 
    "018f5c8b-3d41-7f01-b26c-6693..."
  ],
  "new_title": "Project Summary: Notification Architecture",
  "summary_text": "We finalized the notification architecture using Socket.io for Real-Time and a 30s polling fallback..."
}
```
