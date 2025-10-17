import React, { useEffect, useRef, useState } from 'react'
import { useApp } from '../context/AppContext.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import api from '../services/api.js'
// Lazy-load jsPDF via CDN to avoid bundler resolution issues

export default function Chat() {
  const { currentDocument, chatHistory, setChatHistory, addMessage, currentQuestion, setCurrentQuestion } = useApp()
  const { user } = useAuth()
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const listRef = useRef(null)

  useEffect(() => { listRef.current?.scrollTo(0, listRef.current.scrollHeight) }, [chatHistory])

  // Persistence is handled in AppContext; avoid overwriting freshly loaded history here

  // Ask a suggested question once chat history is hydrated for the selected doc
  const askedRef = useRef(null)
  useEffect(() => {
    if (!currentDocument || !currentQuestion) return
    const key = `${currentDocument.id}:${currentQuestion}`
    if (askedRef.current === key) return
    // Delay slightly to allow AppContext to hydrate chatHistory for this doc
    const t = setTimeout(() => {
      askedRef.current = key
      const q = currentQuestion
      setCurrentQuestion(null)
      ask(q)
    }, 250)
    return () => clearTimeout(t)
  }, [currentQuestion, currentDocument])

  if (!currentDocument) {
    return <div className="card"><h3>No Document Selected</h3><p>Please upload or select a document first.</p></div>
  }

  const ask = async (q) => {
    addMessage('user', q)
    setLoading(true)
    try {
      const data = await api.post('/chat/question', { question: q, document_id: currentDocument.id })
      const answer = data.answer || "Sorry, I couldn't generate an answer."
      const sources = Array.isArray(data.sources) ? data.sources : []
      addMessage('assistant', answer, sources)
    } catch (e) {
      addMessage('assistant', `Sorry, I couldn't process your question. Error: ${e?.response?.data || e.message}`)
    } finally {
      setLoading(false)
    }
  }

  const onSubmit = async (e) => { e.preventDefault(); if (!question.trim()) return; const q = question.trim(); setQuestion(''); await ask(q) }

  return (
    <div className="stack">
      <div className="card">
        <div className="doc-title">{currentDocument.title}</div>
        <div className="muted">Type: {(currentDocument.file_type || 'Unknown').toUpperCase()} • Pages: {currentDocument.pages || 'N/A'} • Words: {currentDocument.word_count || 'N/A'}</div>
      </div>

      <div className="card">
        <h3>Conversation</h3>
        <div className="chat-list" ref={listRef}>
          {chatHistory.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>
              <div className="role">{m.role === 'user' ? 'You' : 'AI'}</div>
              <div className="content">{m.content}</div>
              {Array.isArray(m.sources) && m.sources.length > 0 && (
                <div className="sources">
                  <div className="sources-title">Sources:</div>
                  {m.sources.slice(0,3).map((s, idx) => (
                    <div key={idx} className="source-item">• {(s.text || '').slice(0,100)}...</div>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="muted">Generating answer...</div>}
        </div>
        <form onSubmit={onSubmit} className="row">
          <input placeholder="Ask anything about your document..." value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button disabled={loading || !question.trim()} type="submit">Ask</button>
        </form>
      </div>

      {/* Download buttons at the bottom */}
      <div className="chat-downloads">
        <button className="secondary" type="button" onClick={async () => {
          async function ensureJsPDF(){
            if (window.jspdf?.jsPDF) return window.jspdf.jsPDF
            await new Promise((resolve, reject) => {
              const s = document.createElement('script')
              s.src = 'https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js'
              s.onload = resolve; s.onerror = reject; document.body.appendChild(s)
            })
            return window.jspdf.jsPDF
          }
          const JsPDFCtor = await ensureJsPDF()
          const doc = new JsPDFCtor()
          const lines = chatHistory.map(m => `${m.role === 'user' ? 'You' : 'AI'}: ${m.content}`)
          const pageWidth = doc.internal.pageSize.getWidth() - 20
          let y = 10
          lines.forEach(line => {
            const split = doc.splitTextToSize(line, pageWidth)
            split.forEach(s => { doc.text(s, 10, y); y += 7; if (y > 280){ doc.addPage(); y = 10 } })
          })
          doc.save('smartdocq_chat.pdf')
        }}>Download PDF</button>
        <button className="secondary" type="button" onClick={async () => {
          // Fallback: simple text-based .doc download without docx dependency
          const content = chatHistory.map(m => `${m.role === 'user' ? 'You' : 'AI'}: ${m.content}`).join('\n\n')
          const blob = new Blob([content], { type: 'application/msword' })
          const a = document.createElement('a')
          a.href = URL.createObjectURL(blob)
          a.download = 'smartdocq_chat.doc'
          a.click()
        }}>Download DOC</button>
      </div>
    </div>
  )
}

