import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

export default function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [emailError, setEmailError] = useState('')

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setError(''); setMessage(''); setEmailError('')
    if (!username || !password) { setError('Please enter both username and password.'); return }
    if (!email) { setEmailError('Enter the email. Email is must for registration'); return }
    if (password !== confirm) { setError('Passwords do not match.'); return }
    if (!validateEmail(email)) { setEmailError('Please enter a valid email format'); return }
    setLoading(true)
    const res = await register(username, password, email)
    setLoading(false)
    if (res.ok) { setMessage(res.message || 'Registration successful'); setTimeout(() => navigate('/login'), 800) }
    else setError(res.error)
  }

  return (
    <div className="auth-hero">
      <div className="auth-panel">
        <div className="auth-header">
          <div className="logo">✨</div>
          <h1>Create your account</h1>
          <p className="muted">Start uploading and chatting with documents</p>
        </div>
        <form onSubmit={onSubmit} className="stack">
          <div className="field">
            <label>Username</label>
            <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Choose a username" />
          </div>
          <div className="field">
            <label>Email</label>
            <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="your.email@example.com" />
            {emailError && <div className="error small">{emailError}</div>}
          </div>
          <div className="field">
            <label>Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Create a secure password" />
          </div>
          <div className="field">
            <label>Confirm Password</label>
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="Confirm your password" />
          </div>
          {error && <div className="error">{error}</div>}
          {message && <div className="success">{message}</div>}
          <button type="submit" disabled={loading}>{loading ? 'Creating...' : 'Create Account'}</button>
        </form>
        <div className="muted small">Have an account? <Link to="/login" className="link-highlight">Login</Link></div>
      </div>
    </div>
  )
}

