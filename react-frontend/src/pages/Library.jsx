import React, { useEffect } from 'react'
import { useApp } from '../context/AppContext.jsx'
import { useAuth } from '../context/AuthContext.jsx'

export default function Library() {
  const { documents, setDocuments, currentDocument, setCurrentDocument, setImportantQuestions, setChatHistory } = useApp()
  const { user } = useAuth()

  const remove = (id) => {
    const doc = documents.find(d => d.id === id)
    const rest = documents.filter(d => d.id !== id)
    setDocuments(rest)
    if (currentDocument && currentDocument.id === id) {
      setCurrentDocument(null)
      setImportantQuestions([])
      setChatHistory([])
    }
  }

  useEffect(() => {
    const key = user?.username ? `sdq_recent_${user.username}` : null
    if (!key) return
    try { localStorage.setItem(key, JSON.stringify(documents.slice(0, 10))) } catch {}
  }, [documents, user?.username])

  return (
    <div className="stack">
      <h2 className="page-title">Document Library</h2>
      {documents.length === 0 && <div className="card"><h3>No Documents Found</h3><p>Upload your first document to get started.</p></div>}
      <div className="stack" style={{maxWidth:'1100px',margin:'0 auto', width:'100%'}}>
        {documents.map((doc) => (
          <div key={doc.id} className="card" style={{display:'flex',alignItems:'center',justifyContent:'space-between'}}>
            <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
              <div className="drop-icon" style={{width:40,height:40,fontSize:18}}>📄</div>
              <div>
                <div className="doc-title" style={{marginBottom:4}}>{doc.title}</div>
                <div className="muted">{(doc.file_type||'').toUpperCase()} • {doc.pages} pages • Uploaded {doc.upload_date?.slice(0,10)}</div>
              </div>
            </div>
            <div className="row">
              <button className="secondary" onClick={() => { setCurrentDocument(doc); window.history.pushState({},'', '/upload'); window.dispatchEvent(new PopStateEvent('popstate')) }}>View</button>
              <button className="secondary" onClick={() => remove(doc.id)}>Delete</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

