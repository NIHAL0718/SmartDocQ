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
      // Backend returns: { success, message, token, user_id }
      if (resp && resp.success && resp.token) {
        const u = { username, user_id: resp.user_id || '' }
        const accessToken = resp.token
        setUser(u)
        setToken(accessToken)
        api.setToken(accessToken)
        localStorage.setItem(AUTH_KEY, JSON.stringify({ user: u, token: accessToken }))
        return { ok: true }
      }
      return { ok: false, error: resp?.message || 'Login failed' }
    } catch (e) {
      const msg = e?.response?.data?.detail || e?.message || 'Login failed due to server error'
      return { ok: false, error: msg }
    }
  }

  const register = async (username, password, email) => {
    try {
      const resp = await api.post('/auth/register', { username, password, ...(email ? { email } : {}) })
      // Backend returns: { success, message, token, user_id }
      if (resp && resp.success) {
        return { ok: true, message: resp.message || 'Registration successful' }
      }
      return { ok: false, error: resp?.message || 'Registration failed' }
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








