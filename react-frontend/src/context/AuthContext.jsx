import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import api from '../services/api.js'

const AuthContext = createContext(null)

const AUTH_KEY = 'smartdocq_auth'

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)

  useEffect(() => {
    const stored = localStorage.getItem(AUTH_KEY)
    if (stored) {
      try {
        const { user: u, token: t } = JSON.parse(stored)
        if (t) {
          setUser(u)
          setToken(t)
          api.setToken(t)
        }
      } catch {}
    }
  }, [])

  const isAuthenticated = !!token

  const login = async (username, password) => {
    try {
      const resp = await api.post('/auth/login/json', { username, password })
      // Backend returns: { access_token, token_type, user }
      if (resp && resp.access_token) {
        const u = resp.user || { username }
        const accessToken = resp.access_token
        setUser(u)
        setToken(accessToken)
        api.setToken(accessToken)
        localStorage.setItem(AUTH_KEY, JSON.stringify({ user: u, token: accessToken }))
        return { ok: true }
      }
      return { ok: false, error: resp?.message || resp?.detail || 'Login failed' }
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || 'Login failed due to server error'
      return { ok: false, error: msg }
    }
  }

  const register = async (username, password, email) => {
    try {
      const resp = await api.post('/auth/register', { username, password, ...(email ? { email } : {}) })
      // On success backend returns created user object
      if (resp && (resp.id || resp.username)) {
        return { ok: true, message: 'Registration successful' }
      }
      return { ok: false, error: resp?.message || resp?.detail || 'Registration failed' }
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || 'Registration failed'
      return { ok: false, error: msg }
    }
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    api.clearToken()
    localStorage.removeItem(AUTH_KEY)
  }

  const value = useMemo(() => ({ user, token, isAuthenticated, login, register, logout }), [user, token, isAuthenticated])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}








