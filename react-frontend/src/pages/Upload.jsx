import React, { useEffect, useState } from 'react'
import api from '../services/api.js'
import { useApp } from '../context/AppContext.jsx'
import { useNavigate } from 'react-router-dom'

export default function Upload() {
  const { documents, setDocuments, setCurrentDocument, setImportantQuestions, importantQuestions, setCurrentQuestion, currentDocument, persistImportantQuestionsForCurrent, addToRecent } = useApp()
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const onSubmit = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true); setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('title', title || file.name)
      form.append('language', 'english')
      const upload = await api.post('/documents/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      const docId = upload.id
      const newDoc = {
        id: docId,
        title: upload.title || (title || file.name),
        upload_date: new Date().toISOString().slice(0,19),
        file_type: (file.name.split('.').pop() || '').toLowerCase(),
        language: upload.language || 'english',
        pages: upload.page_count || 0,
        chunks: upload.chunk_count || 0,
        word_count: upload.word_count || 0,
      }
      setCurrentDocument(newDoc)
      setDocuments((prev) => [newDoc, ...prev])
      addToRecent(newDoc)

      // poll status up to ~12s similar to Streamlit
      for (let i = 0; i < 8; i++) {
        try {
          const status = await api.get(`/documents/status/${docId}`)
          if (status) {
            newDoc.pages = status.page_count ?? newDoc.pages
            newDoc.chunks = status.chunk_count ?? newDoc.chunks
            newDoc.word_count = status.word_count ?? newDoc.word_count
            if (status.status === 'completed') break
          }
        } catch {}
        await new Promise(r => setTimeout(r, 1500))
      }

      try {
        const q = await api.get(`/documents/${docId}/questions`)
        const qs = q.questions || []
        setImportantQuestions(qs)
        persistImportantQuestionsForCurrent(qs)
      } catch {
        const qs = [
          'What are the main themes discussed in the document?',
          'How does the document address the key challenges?',
          'What solutions are proposed in the document?',
          'Who are the main stakeholders mentioned?',
          'What are the potential implications of the findings?',
        ]
        setImportantQuestions(qs)
        persistImportantQuestionsForCurrent(qs)
      }
    } catch (err) {
      setError(err?.response?.data || err?.message || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const fileInputRef = React.useRef(null)
  const onBrowseClick = () => fileInputRef.current?.click()
  const onDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer?.files?.[0]) setFile(e.dataTransfer.files[0])
  }
  const onDragOver = (e) => { e.preventDefault() }
  const stopDrag = (e) => { e.stopPropagation() }

  return (
    <div className="stack" style={{width:'100%'}}>
      <h2 className="page-title">Upload Document</h2>
      <div className="card upload-card" style={{width:'100%'}}>
        <div className="section-title">Upload a document</div>
        <div className="dropzone" onDrop={onDrop} onDragOver={onDragOver} onClick={onBrowseClick} onTouchStart={stopDrag}>
          <div className="drop-icon">⬆</div>
          <div className="drop-title">Drag and drop file here</div>
          <div className="drop-hint">Limit 200MB per file • PDF, DOCX, TXT, JPG, JPEG, PNG</div>
          <button type="button" className="btn-secondary" style={{position:'relative', zIndex:1}} onClick={(e) => { e.stopPropagation(); onBrowseClick() }}>Browse files</button>
          <input ref={fileInputRef} type="file" className="hidden-input" accept=".pdf,.docx,.txt,.jpg,.jpeg,.png" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          {file && <div className="file-name">Selected: {file.name}</div>}
        </div>

        <div className="field" style={{ marginTop: 18 }}>
          <label>Document Title (optional)</label>
          <input placeholder="Enter document title..." value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>

        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

        <button className="btn-primary-large" style={{ marginTop: 12 }} type="button" onClick={onSubmit} disabled={loading || !file}>
          {loading ? 'Uploading...' : 'Upload'}
        </button>
      </div>

      {currentDocument && (
        <div className="card">
          <h3>Current Document</h3>
          <div className="grid two">
            <div className="stat"><div className="stat-value">{currentDocument.pages||'N/A'}</div><div className="stat-label">Pages</div></div>
            <div className="stat"><div className="stat-value">{currentDocument.word_count||'N/A'}</div><div className="stat-label">Words</div></div>
          </div>
          <div className="muted" style={{marginTop:8}}>Uploaded on {currentDocument.upload_date?.slice(0,10) || 'N/A'} • Type {String(currentDocument.file_type||'').toUpperCase()}</div>
        </div>
      )}

      {importantQuestions && importantQuestions.length > 0 && (
        <div className="card">
          <h3>Suggested Questions</h3>
          <div className="pills">
            {importantQuestions.map((q, i) => (
              <button key={i} type="button" className="pill" onClick={() => {
                setCurrentQuestion(q)
                navigate('/chat')
              }}>{q}</button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

