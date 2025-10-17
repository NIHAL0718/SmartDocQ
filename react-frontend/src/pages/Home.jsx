import React, { useEffect, useMemo, useState } from 'react'
import { useApp } from '../context/AppContext.jsx'
import { useAuth } from '../context/AuthContext.jsx'

export default function Home() {
  const { documents, chatHistory, setCurrentDocument } = useApp()
  const { user } = useAuth()
  const [recent, setRecent] = useState([])

  useEffect(() => {
    const key = user?.username ? `sdq_recent_${user.username}` : null
    if (!key) { setRecent([]); return }
    try { setRecent(JSON.parse(localStorage.getItem(key) || '[]')) } catch { setRecent([]) }
  }, [user?.username, documents])
  const stats = useMemo(() => {
    const totalDocs = documents.length
    const totalPages = documents.reduce((s, d) => s + (d.pages || 0), 0)
    const totalWords = documents.reduce((s, d) => s + (d.word_count || 0), 0)
    return { totalDocs, totalPages, totalWords, chatMessages: chatHistory.length }
  }, [documents, chatHistory])

  return (
    <div className="stack">
      <section className="hero card">
        <h1 className="hero-title">Chat with your documents</h1>
        <p className="hero-sub">Upload PDFs, run OCR, translate content, and ask questions — all in one place.</p>
        <div className="row">
          <a className="nav-link fancy" href="/upload" onClick={(e)=>{e.preventDefault();window.history.pushState({},'', '/upload');window.dispatchEvent(new PopStateEvent('popstate'))}}><span>Upload a Document</span></a>
          <a className="nav-link fancy" href="/chat" onClick={(e)=>{e.preventDefault();window.history.pushState({},'', '/chat');window.dispatchEvent(new PopStateEvent('popstate'))}}><span>Open Chat</span></a>
        </div>
      </section>

      <div className="grid three">
        <div className="card stat"><div className="stat-value">{stats.totalDocs}</div><div className="stat-label">Documents</div></div>
        <div className="card stat"><div className="stat-value">{stats.totalPages}</div><div className="stat-label">Pages</div></div>
        <div className="card stat"><div className="stat-value">{stats.totalWords}</div><div className="stat-label">Words</div></div>
      </div>

      <section className="grid two">
        <div className="card">
          <h3>Upload & Process</h3>
          <p className="muted">Drag-and-drop your documents. We’ll extract, chunk, and index them for lightning-fast answers.</p>
        </div>
        <div className="card">
          <h3>OCR & Translate</h3>
          <p className="muted">Use OCR for images and scanned PDFs. Translate content across multiple languages.</p>
        </div>
      </section>

      {(recent.length > 0 || documents.length > 0) && (
        <div className="stack">
          <h3>Recent Documents</h3>
          <div className="grid three">
            {(recent.length ? recent : documents).slice(0,3).map((doc) => (
              <div key={doc.id} className="card" style={{cursor:'pointer'}} onClick={(e)=>{e.preventDefault(); setCurrentDocument(doc); window.history.pushState({},'', '/upload'); window.dispatchEvent(new PopStateEvent('popstate'))}}>
                <div className="doc-title">{doc.title}</div>
                <div className="muted">{(doc.file_type || '').toUpperCase()} • {doc.pages} pages • {doc.word_count} words</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

