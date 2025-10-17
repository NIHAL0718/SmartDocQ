import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const onSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    const res = await login(username, password)
    setLoading(false)
    if (res.ok) navigate('/')
    else setError(res.error)
  }

  return (
    <div className="auth-hero">
      <div className="auth-panel">
        <div className="auth-header">
          <div className="logo">📚</div>
          <h1>Welcome back</h1>
          <p className="muted">Log in to manage and chat with your documents</p>
        </div>
        <form onSubmit={onSubmit} className="stack">
          <div className="field">
            <label>Username</label>
            <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Enter your username" />
          </div>
          <div className="field">
            <label>Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter your password" />
          </div>
          {error && <div className="error">{error}</div>}
          <button type="submit" disabled={loading}>{loading ? 'Logging in...' : 'Login'}</button>
        </form>
        <div className="muted small">No account? <Link to="/register" className="link-highlight">Create one</Link></div>
      </div>
    </div>
  )
}


