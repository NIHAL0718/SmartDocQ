import React, { useEffect, useState } from 'react'
import api from '../services/api.js'
import { useApp } from '../context/AppContext.jsx'
import { useAuth } from '../context/AuthContext.jsx'

export default function OCR() {
  const { currentDocument } = useApp()
  const { user } = useAuth()
  const [file, setFile] = useState(null)
  const [enhance, setEnhance] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const ocrKeyForDoc = (docId) => (user?.username && docId ? `sdq_ocr_${user.username}_${docId}` : null)

  // Load saved OCR result for current document
  useEffect(() => {
    if (!currentDocument?.id) return
    const key = ocrKeyForDoc(currentDocument.id)
    try { const saved = JSON.parse(localStorage.getItem(key) || 'null'); if (saved) setResult(saved) } catch {}
  }, [currentDocument?.id])

  const onSubmit = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true); setError(''); setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('enhance_image', String(enhance))
      const data = await api.post('/ocr/process', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResult(data)
      if (currentDocument?.id) {
        const key = ocrKeyForDoc(currentDocument.id)
        try { localStorage.setItem(key, JSON.stringify(data)) } catch {}
      }
    } catch (e) {
      setError(e?.response?.data || e.message || 'OCR failed')
    } finally { setLoading(false) }
  }

  const copyText = () => { if (result?.text) navigator.clipboard.writeText(result.text) }
  const downloadText = () => { if (!result?.text) return; const blob = new Blob([result.text], { type: 'text/plain' }); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'ocr_result.txt'; a.click() }

  return (
    <div className="stack">
      <h2 className="page-title">Optical Character Recognition (OCR)</h2>
      <div className="card upload-card">
        <div className="section-title">Upload an image or scanned document</div>
        <div className="dropzone" onClick={() => document.getElementById('ocrInput').click()}>
          <div className="drop-icon">⬆</div>
          <div className="drop-title">Drag and drop file here</div>
          <div className="drop-hint">Limit 200MB per file • JPG, JPEG, PNG, PDF</div>
          <button type="button" className="btn-secondary" onClick={(e)=>{e.stopPropagation(); document.getElementById('ocrInput').click()}}>Browse files</button>
          <input id="ocrInput" className="hidden-input" type="file" accept=".jpg,.jpeg,.png,.pdf" onChange={(e)=> setFile(e.target.files?.[0] || null)} />
        </div>
        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
        <button className="btn-primary-large" style={{ marginTop: 12 }} disabled={loading || !file} onClick={onSubmit}>{loading ? 'Processing...' : 'Process with OCR'}</button>
      </div>

      {result && (
        <div className="card" style={{maxWidth:'1200px',margin:'0 auto', width:'100%'}}>
          <h3>Extracted Text</h3>
          {result.text ? (
            <textarea value={result.text} readOnly rows={18} />
          ) : (
            <div className="muted">No text could be extracted from this image.</div>
          )}
          <div className="row" style={{justifyContent:'flex-end'}}>
            <button className="secondary" type="button" onClick={copyText}>Copy</button>
            <button className="secondary" type="button" onClick={downloadText}>Download .txt</button>
          </div>
        </div>
      )}
    </div>
  )
}


