import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import api from '../services/api.js'
import { useAuth } from './AuthContext.jsx'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [documents, setDocuments] = useState([])
  const [currentDocument, setCurrentDocument] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [importantQuestions, setImportantQuestions] = useState([])
  const [currentQuestion, setCurrentQuestion] = useState(null)
  const { token, user } = useAuth()
  
  // Persist and restore per-user, per-document state (questions, chat)
  const getUserKey = (suffix) => (user?.username ? `sdq_${suffix}_${user.username}` : null)
  const questionsKeyForDoc = (docId) => (user?.username && docId ? `sdq_questions_${user.username}_${docId}` : null)
  const chatKeyForDoc = (docId) => (user?.username && docId ? `sdq_chat_${user.username}_${docId}` : null)

  // Load cached documents immediately on login (per user), then refresh from backend
  useEffect(() => {
    if (!token) { setDocuments([]); return }
    const userKey = user?.username ? `sdq_docs_${user.username}` : null
    if (userKey) {
      try { const cached = JSON.parse(localStorage.getItem(userKey) || '[]'); if (Array.isArray(cached)) setDocuments(cached) } catch {}
    }
    api.get('/documents/list').then((items) => {
      if (Array.isArray(items)) {
        const mapped = items.map((it) => ({
          id: it.id,
          title: it.title,
          upload_date: it.upload_date,
          file_type: it.file_type || 'unknown',
          language: it.language || 'english',
          pages: it.page_count ?? it.pages ?? 0,
          chunks: it.chunk_count ?? it.chunks ?? 0,
          word_count: it.word_count ?? 0,
        }))
        setDocuments(mapped)
        // Enrich with latest status to populate pages/words counts
        Promise.all(mapped.map(async (d) => {
          try {
            const s = await api.get(`/documents/status/${d.id}`)
            return { id: d.id, pages: s.page_count ?? d.pages, word_count: s.word_count ?? d.word_count, chunks: s.chunk_count ?? d.chunks }
          } catch { return null }
        })).then((updates) => {
          const u = updates.filter(Boolean)
          if (u.length) {
            setDocuments((prev) => prev.map((doc) => {
              const m = u.find(x => x.id === doc.id)
              return m ? { ...doc, pages: m.pages, word_count: m.word_count, chunks: m.chunks } : doc
            }))
          }
        }).catch(() => {})
      }
    }).catch(() => {})
  }, [token, user?.username])

  // Persist documents per user to localStorage for resilience across server restarts
  useEffect(() => {
    const userKey = user?.username ? `sdq_docs_${user.username}` : null
    if (userKey) {
      try { localStorage.setItem(userKey, JSON.stringify(documents)) } catch {}
    }
  }, [documents, user?.username])

  // When selecting a current document, restore its important questions and chat from storage
  useEffect(() => {
    if (!currentDocument) return
    // Clear previous doc state immediately to avoid showing stale data
    setImportantQuestions([])
    setChatHistory([])
    const qKey = questionsKeyForDoc(currentDocument.id)
    try {
      if (qKey) {
        const savedQ = JSON.parse(localStorage.getItem(qKey) || 'null')
        if (Array.isArray(savedQ) && savedQ.length) setImportantQuestions(savedQ)
      }
    } catch {}
    const cKey = chatKeyForDoc(currentDocument.id)
    try {
      if (cKey) {
        const savedC = JSON.parse(localStorage.getItem(cKey) || 'null')
        if (Array.isArray(savedC)) setChatHistory(savedC)
      }
    } catch {}
    // If questions not found in storage, fetch from backend
    api.get(`/documents/${currentDocument.id}/questions`).then((q) => {
      const qs = q?.questions || []
      if (qs.length) {
        setImportantQuestions(qs)
        persistImportantQuestionsForCurrent(qs)
      }
    }).catch(() => {})
    // Refresh status to ensure stats are up to date
    api.get(`/documents/status/${currentDocument.id}`).then((s) => {
      const updated = { ...currentDocument, pages: s.page_count ?? currentDocument.pages, chunks: s.chunk_count ?? currentDocument.chunks, word_count: s.word_count ?? currentDocument.word_count }
      setCurrentDocument(updated)
      setDocuments((prev) => prev.map(d => d.id === updated.id ? updated : d))
    }).catch(() => {})
  }, [currentDocument?.id])

  // Persist chat history per current document
  useEffect(() => {
    if (!currentDocument?.id) return
    const cKey = chatKeyForDoc(currentDocument.id)
    if (cKey) {
      try { localStorage.setItem(cKey, JSON.stringify(chatHistory)) } catch {}
    }
  }, [chatHistory, currentDocument?.id])

  // Helpers to persist questions per doc and maintain a per-user recent list
  const persistImportantQuestionsForCurrent = (questions) => {
    if (!currentDocument?.id) return
    const qKey = questionsKeyForDoc(currentDocument.id)
    if (qKey) {
      try { localStorage.setItem(qKey, JSON.stringify(questions || [])) } catch {}
    }
  }

  const addToRecent = (doc) => {
    const key = getUserKey('recent')
    if (!key || !doc) return
    try {
      const existing = JSON.parse(localStorage.getItem(key) || '[]')
      const filtered = [doc, ...existing.filter(d => d.id !== doc.id)].slice(0, 10)
      localStorage.setItem(key, JSON.stringify(filtered))
    } catch {}
  }

  const addMessage = (role, content, sources) => {
    setChatHistory((prev) => [...prev, { role, content, timestamp: Date.now() / 1000, ...(sources ? { sources } : {}) }])
  }

  const value = useMemo(() => ({
    documents, setDocuments,
    currentDocument, setCurrentDocument,
    chatHistory, setChatHistory,
    importantQuestions, setImportantQuestions,
    currentQuestion, setCurrentQuestion,
    addMessage,
    persistImportantQuestionsForCurrent,
    addToRecent,
  }), [documents, currentDocument, chatHistory, importantQuestions, currentQuestion])

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() { return useContext(AppContext) }



