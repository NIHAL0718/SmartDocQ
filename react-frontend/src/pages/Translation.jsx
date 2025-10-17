import React, { useEffect, useMemo, useRef, useState } from 'react'
import api from '../services/api.js'
import { useApp } from '../context/AppContext.jsx'

// Backend-based TTS controller with Stop toggle and graceful fallback
const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api')
function useBackendTTSController(){
  const [isPlaying, setIsPlaying] = useState(false)
  const abortRef = useRef(false)
  const audioRef = useRef(null)

  const stop = () => {
    abortRef.current = true
    try { if (audioRef.current){ audioRef.current.pause(); audioRef.current.currentTime = 0 } } catch {}
    try { window.speechSynthesis && window.speechSynthesis.cancel() } catch {}
    setIsPlaying(false)
  }

  const play = async (text, languageCode) => {
    if (!text) return
    // If already playing, interpret as stop toggle
    if (isPlaying) { stop(); return }
    abortRef.current = false
    setIsPlaying(true)
    try {
      // Clean and cap length to make start immediate
      const cleanedFull = String(text).replace(/https?:\/\/\S+/g, ' ').replace(/\s+/g, ' ').trim()
      const maxLen = 6000
      const cleaned = cleanedFull.length > maxLen ? cleanedFull.slice(0, maxLen) : cleanedFull
      const lang = (languageCode||'en').toLowerCase()
      // Use a small head chunk to start audio faster, then larger chunks
      const chunks = []
      const head = cleaned.slice(0, Math.min(700, cleaned.length))
      if (head) chunks.push(head)
      let idx = head.length
      while (idx < cleaned.length) { chunks.push(cleaned.slice(idx, idx + 3000)); idx += 3000 }
      for (const part of chunks){
        if (abortRef.current) break
        const resp = await fetch(`${API_BASE}/translation/tts`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: part, language: lang })
        })
        if (!resp.ok) throw new Error(`TTS request failed: ${resp.status}`)
        const blob = await resp.blob()
        const url = URL.createObjectURL(blob)
        await new Promise((resolve) => {
          const a = new Audio(url)
          audioRef.current = a
          a.onended = resolve
          a.onerror = resolve
          a.play().catch(resolve)
        })
      }
    } catch (e) {
      // Fallback to browser SpeechSynthesis if backend TTS fails (e.g., 404)
      try {
        const utter = new SpeechSynthesisUtterance(String(text))
        utter.lang = languageCode || 'en'
        utter.onstart = () => { setIsPlaying(true) }
        utter.onend = () => { setIsPlaying(false) }
        utter.onerror = () => { setIsPlaying(false) }
        try { window.speechSynthesis?.cancel() } catch {}
        window.speechSynthesis?.speak(utter)
        return
      } catch {}
    }
    setIsPlaying(false)
  }

  return { isPlaying, play, stop }
}

export default function Translation() {
  const { currentDocument } = useApp()
  const [languages, setLanguages] = useState([])
  const [tab, setTab] = useState('text')
  const [history, setHistory] = useState([])

  useEffect(() => {
    api.get('/translation/languages').then((res) => {
      if (Array.isArray(res.languages)) setLanguages(res.languages)
      else if (Array.isArray(res)) setLanguages(res)
    }).catch(() => setLanguages(defaultLangs))
    // Only load the History here; do not restore per-tab working state automatically
    const saved = localStorage.getItem('sdq_translation_history')
    if (saved) try { setHistory(JSON.parse(saved)) } catch {}
  }, [])

  const pushHistory = (entry) => {
    // Always read the latest from localStorage to avoid resurrecting cleared items
    let current = []
    try { current = JSON.parse(localStorage.getItem('sdq_translation_history') || '[]') } catch {}
    const next = [...current, { ...entry, ts: Date.now() }]
    localStorage.setItem('sdq_translation_history', JSON.stringify(next))
    setHistory(next)
  }

  return (
    <div className="stack">
      <h2>Translation Service</h2>
      <div className="tabs">
        <button className={tab==='text'?'active':''} onClick={() => setTab('text')}>Text Translation</button>
        <button className={tab==='current'?'active':''} onClick={() => setTab('current')}>Current Document</button>
        <button className={tab==='upload'?'active':''} onClick={() => setTab('upload')}>Upload Document</button>
      </div>

      {tab === 'text' && <TextTranslation languages={languages} pushHistory={pushHistory} />}
      {tab === 'current' && <CurrentDocumentTranslation languages={languages} currentDocument={currentDocument} pushHistory={pushHistory} />}
      {tab === 'upload' && <UploadDocumentTranslation languages={languages} pushHistory={pushHistory} />}
    </div>
  )
}

// -------- TTS utilities ---------
function useTTS() {
  const speakingRef = useRef(false)
  const [voices, setVoices] = useState([])
  const [isSpeaking, setIsSpeaking] = useState(false)
  
  useEffect(() => {
    const loadVoices = () => {
      const availableVoices = window.speechSynthesis?.getVoices?.() || []
      setVoices(availableVoices)
    }
    
    loadVoices()
    if (window.speechSynthesis) {
      window.speechSynthesis.addEventListener('voiceschanged', loadVoices)
      return () => window.speechSynthesis.removeEventListener('voiceschanged', loadVoices)
    }
  }, [])
  
  const pickVoice = (langCode) => {
    if (!voices.length) return null
    
    const code = (langCode || '').toLowerCase()
    console.log('Picking voice for language code:', code)
    console.log('Available voices:', voices.map(v => ({ name: v.name, lang: v.lang })))
    
    // Try exact language match first
    let voice = voices.find(v => v.lang === code)
    console.log('Exact match result:', voice?.name || 'none')
    
    // Try language family match
    if (!voice) {
      const langFamily = code.split('-')[0]
      voice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith(langFamily.toLowerCase()))
      console.log('Family match result:', voice?.name || 'none')
    }
    
    // Try name-based matching for specific languages
    if (!voice) {
      if (code === 'te' || code.startsWith('te')) {
        // Telugu - try multiple variations
        voice = voices.find(v => {
          const name = (v.name || '').toLowerCase()
          const lang = (v.lang || '').toLowerCase()
          return name.includes('telugu') || 
                 name.includes('telegu') || 
                 name.includes('telugu') ||
                 lang === 'te-in' || 
                 lang === 'te' ||
                 lang.startsWith('te-')
        })
        console.log('Telugu name match result:', voice?.name || 'none')
      } else if (code === 'ta' || code.startsWith('ta')) {
        // Tamil
        voice = voices.find(v => {
          const name = (v.name || '').toLowerCase()
          const lang = (v.lang || '').toLowerCase()
          return name.includes('tamil') || 
                 lang === 'ta-in' || 
                 lang === 'ta' ||
                 lang.startsWith('ta-')
        })
        console.log('Tamil name match result:', voice?.name || 'none')
      }
    }
    
    // Fallback to default or first available
    const result = voice || voices.find(v => v.default) || voices[0]
    console.log('Final selected voice:', result?.name, 'lang:', result?.lang)
    return result
  }
  
  const speak = (text, langCode, onEnd, attempt = 0) => {
    const synth = window.speechSynthesis
    if (!synth || !text) return false

    // Wait for voices to be ready if needed
    if ((!voices || voices.length === 0) && attempt < 3) {
      setTimeout(() => speak(text, langCode, onEnd, attempt + 1), 250)
      return true
    }

    if (synth.speaking) synth.cancel()
    try { synth.resume && synth.resume() } catch {}

    const utter = new SpeechSynthesisUtterance(text)
    const voice = pickVoice(langCode)
    
    if (voice) {
      utter.voice = voice
      utter.lang = voice.lang
      console.log('Using voice:', voice.name, 'with lang:', voice.lang)
    } else if (langCode) {
      // Force script-specific BCP-47 if common mapping isn't present
      const c = (langCode||'').toLowerCase()
      if (c.startsWith('te')) {
        utter.lang = 'te-IN' // Telugu
        console.log('No Telugu voice found, using te-IN language code')
      } else if (c.startsWith('ta')) {
        utter.lang = 'ta-IN' // Tamil
        console.log('No Tamil voice found, using ta-IN language code')
      } else {
        utter.lang = langCode
        console.log('Using language code:', langCode)
      }
    }
    

    utter.rate = 1
    utter.pitch = 1
    utter.volume = 1
    
    utter.onstart = () => {
      speakingRef.current = true
      setIsSpeaking(true)
    }
    utter.onend = () => { 
      speakingRef.current = false
      setIsSpeaking(false)
      onEnd && onEnd() 
    }
    utter.onerror = () => {
      speakingRef.current = false
      setIsSpeaking(false)
      console.warn('Speech synthesis error')
    }
    
    synth.speak(utter)
    return true
  }
  
  const stop = () => { 
    try { 
      window.speechSynthesis?.cancel()
      speakingRef.current = false 
      setIsSpeaking(false)
    } catch {} 
  }

  return { speak, stop, speakingRef, isSpeaking, voices }
}

function TextTranslation({ languages, pushHistory }) {
  // Do not preload previous text/result on first load; start fresh each session
  const [text, setText] = useState('')
  const [source, setSource] = useState('Auto Detect')
  const [target, setTarget] = useState(languages[0]?.name || 'English')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const map = useMemo(() => Object.fromEntries(languages.map(l => [l.name, l.code])), [languages])
  const { isPlaying, play, stop } = useBackendTTSController()

  // No persistence for working state here; only history is persisted

  const translate = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true); setResult(null)
    const payload = { text, target_language: map[target] }
    if (source !== 'Auto Detect') payload.source_language = map[source]
    try {
      const data = await api.post('/translation/translate', payload)
      setResult(data)
      pushHistory({ type:'text', text, result: data.translated_text, src: source, tgt: target })
    } catch (e) {
      setResult({ translated_text: text, error: e?.response?.data || e.message })
    } finally { setLoading(false) }
  }

  return (
    <div className="card">
      <form onSubmit={translate} className="stack">
        <textarea rows={6} placeholder="Enter text to translate" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="row">
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option>Auto Detect</option>
            {languages.map(l => <option key={l.code}>{l.name}</option>)}
          </select>
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            {languages.map(l => <option key={l.code}>{l.name}</option>)}
          </select>
          <button disabled={loading} type="submit">{loading ? 'Translating...' : 'Translate'}</button>
        </div>
      </form>
      {result && (
        <div className="stack">
          <h3>Translation Result</h3>
          <textarea rows={18} value={result.translated_text || ''} readOnly />
          <div className="row" style={{justifyContent:'flex-end'}}>
            <button className="secondary" type="button" onClick={() => {
              const a = document.createElement('a')
              const blob = new Blob([result.translated_text||''], { type:'text/plain' })
              a.href = URL.createObjectURL(blob)
              a.download = 'translation.txt'
              a.click()
            }}>Download .txt</button>
            {/* Use backend TTS for reliable Telugu/Tamil; keep browser fallback in playTTS */}
            <button className="secondary" type="button" onClick={() => play(result.translated_text||'', map[target])}>{isPlaying ? 'Stop' : 'Listen'}</button>
          </div>
        </div>
      )}
      
      
      <TranslationHistory />
    </div>
  )
}

function CurrentDocumentTranslation({ languages, currentDocument, pushHistory }) {
  // Only keep per-session state; do not restore previous document results unless same doc is selected and stored
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [src, setSrc] = useState('Auto Detect')
  const [tgt, setTgt] = useState(languages[0]?.name || 'English')
  const map = useMemo(() => Object.fromEntries(languages.map(l => [l.name, l.code])), [languages])
  const { isPlaying, play, stop } = useBackendTTSController()


  const translate = async () => {
    if (!currentDocument) return
    setLoading(true); setResult(null)
    try {
      const content = await api.get(`/documents/${currentDocument.id}/content`)
      const text = content.content || ''
      const payload = { text, target_language: map[tgt] }
      if (src !== 'Auto Detect') payload.source_language = map[src]
      const data = await api.post('/translation/translate', payload)
      setResult(data)
      pushHistory({ type:'current', filename: currentDocument.title, text, result: data.translated_text, src, tgt })
    } catch (e) {
      setResult({ translated_text: '', error: e?.response?.data || e.message })
    } finally { setLoading(false) }
  }

  if (!currentDocument) return <div className="card">No Document Selected</div>

  return (
    <div className="card">
      <div className="row">
        <select value={src} onChange={(e) => setSrc(e.target.value)}>
          <option>Auto Detect</option>
          {languages.map(l => <option key={l.code}>{l.name}</option>)}
        </select>
        <select value={tgt} onChange={(e) => setTgt(e.target.value)}>
          {languages.map(l => <option key={l.code}>{l.name}</option>)}
        </select>
        <button onClick={translate} disabled={loading}>{loading ? 'Translating...' : 'Translate Document'}</button>
      </div>
      {result && (
        <div className="stack">
          <h3>Translation Result</h3>
          <textarea rows={20} value={result.translated_text || ''} readOnly />
          <div className="row" style={{justifyContent:'flex-end'}}>
            <button className="secondary" type="button" onClick={() => {
              const a = document.createElement('a')
              const blob = new Blob([result.translated_text||''], { type:'text/plain' })
              a.href = URL.createObjectURL(blob)
              a.download = `translated_${(currentDocument?.title||'document')}.txt`
              a.click()
            }}>Download .txt</button>
            <button className="secondary" type="button" onClick={() => play(result.translated_text||'', map[tgt])}>{isPlaying ? 'Stop' : 'Listen'}</button>
          </div>
        </div>
      )}
    </div>
  )
}

function UploadDocumentTranslation({ languages, pushHistory }) {
  // Do not preload previous upload state on re-visit
  const [file, setFile] = useState(null)
  const [src, setSrc] = useState('Auto Detect')
  const [tgt, setTgt] = useState(languages[0]?.name || 'English')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const map = useMemo(() => Object.fromEntries(languages.map(l => [l.name, l.code])), [languages])
  const { isPlaying, play, stop } = useBackendTTSController()


  const translate = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true); setResult(null)
    // Extract text client-side for common types similar to Streamlit
    try {
      const ext = file.name.split('.').pop().toLowerCase()
      let text = ''
      if (ext === 'txt') {
        text = await file.text()
      } else if (ext === 'pdf' || ext === 'docx') {
        // Upload to backend then fetch content, keeping endpoints unchanged
        const form = new FormData()
        form.append('file', file)
        form.append('title', file.name)
        form.append('language', 'english')
        const uploaded = await api.post('/documents/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
        const docId = uploaded.id
        const content = await api.get(`/documents/${docId}/content`)
        text = content.content || ''
      }
      const payload = { text, target_language: map[tgt] }
      if (src !== 'Auto Detect') payload.source_language = map[src]
      const data = await api.post('/translation/translate', payload)
      setResult(data)
      pushHistory({ type:'upload', filename: file.name, text, result: data.translated_text, src, tgt })
    } catch (e) {
      setResult({ translated_text: '', error: e?.response?.data || e.message })
    } finally { setLoading(false) }
  }

  return (
    <div className="card upload-card">
      <form onSubmit={translate} className="stack">
        <div className="dropzone" onClick={() => document.getElementById('transUpload').click()}>
          <div className="drop-icon">⬆</div>
          <div className="drop-title">Drag and drop file here</div>
          <div className="drop-hint">PDF, DOCX, TXT</div>
          <button type="button" className="btn-secondary" onClick={(e)=>{e.preventDefault(); e.stopPropagation(); document.getElementById('transUpload').click()}}>Browse files</button>
          <input id="transUpload" className="hidden-input" type="file" accept=".pdf,.docx,.txt" onChange={(e)=> setFile(e.target.files?.[0] || null)} />
          {file && <div className="file-name">Selected: {file.name}</div>}
        </div>
        <div className="row">
          <select value={src} onChange={(e) => setSrc(e.target.value)}>
            <option>Auto Detect</option>
            {languages.map(l => <option key={l.code}>{l.name}</option>)}
          </select>
          <select value={tgt} onChange={(e) => setTgt(e.target.value)}>
            {languages.map(l => <option key={l.code}>{l.name}</option>)}
          </select>
          <button disabled={loading || !file} type="submit">{loading ? 'Translating...' : 'Translate Document'}</button>
        </div>
      </form>
      {result && (
        <div className="stack">
          <h3>Translation Result</h3>
          <textarea rows={20} value={result.translated_text || ''} readOnly />
          <div className="row" style={{justifyContent:'flex-end'}}>
            <button className="secondary" type="button" onClick={() => {
              const a = document.createElement('a')
              const blob = new Blob([result.translated_text||''], { type:'text/plain' })
              a.href = URL.createObjectURL(blob)
              a.download = `translated_${(file?.name||'document')}.txt`
              a.click()
            }}>Download .txt</button>
            <button className="secondary" type="button" onClick={() => play(result.translated_text||'', map[tgt])}>{isPlaying ? 'Stop' : 'Listen'}</button>
          </div>
        </div>
      )}
    </div>
  )
}

function TranslationHistory() {
  const [items, setItems] = useState(() => { try { return JSON.parse(localStorage.getItem('sdq_translation_history') || '[]') } catch { return [] } })
  const [filter, setFilter] = useState('all')
  const refresh = () => { try { setItems(JSON.parse(localStorage.getItem('sdq_translation_history') || '[]')) } catch {} }
  const clearSection = (type) => {
    const next = (items||[]).filter(i => i.type !== type)
    localStorage.setItem('sdq_translation_history', JSON.stringify(next)); setItems(next)
  }
  const delItem = (ts) => { const next = (items||[]).filter(i => i.ts !== ts); localStorage.setItem('sdq_translation_history', JSON.stringify(next)); setItems(next) }
  useEffect(() => { const id = setInterval(refresh, 1000); return () => clearInterval(id) }, [])
  if (!items.length) return null
  const visible = items.filter(i => filter==='all' ? true : i.type===filter).slice().reverse()
  return (
    <div className="card">
      <div className="row" style={{justifyContent:'space-between',alignItems:'center'}}>
        <h3 style={{margin:0}}>History</h3>
        <div className="row">
          <select value={filter} onChange={(e)=>setFilter(e.target.value)}>
            <option value="all">All</option>
            <option value="text">Text</option>
            <option value="current">Current Document</option>
            <option value="upload">Upload</option>
          </select>
          {filter!=='all' && <button className="secondary" onClick={()=>clearSection(filter)}>Clear {filter}</button>}
          {filter==='all' && <button className="secondary" onClick={()=>{ localStorage.removeItem('sdq_translation_history'); setItems([]) }}>Clear All</button>}
        </div>
      </div>
      <div className="stack">
        {visible.map((it, idx) => (
          <div key={it.ts || idx} className="card" style={{padding:'12px'}}>
            <div className="row" style={{justifyContent:'space-between',alignItems:'center'}}>
              <div className="muted">{(it.type||'text').toUpperCase()} • {it.src} → {it.tgt}</div>
              <button className="secondary" onClick={()=>delItem(it.ts)}>Delete</button>
            </div>
            {it.filename && <div className="muted" style={{marginTop:'6px'}}>File: {it.filename}</div>}
            <div className="muted" style={{marginTop:'6px'}}>Text: {String(it.text||'').slice(0,160)}{String(it.text||'').length>160?'…':''}</div>
            <div style={{marginTop:'6px'}}>Result: {String(it.result||'').slice(0,240)}{String(it.result||'').length>240?'…':''}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

const defaultLangs = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
  { code: 'bn', name: 'Bengali' },
  { code: 'te', name: 'Telugu' },
  { code: 'ta', name: 'Tamil' },
]

