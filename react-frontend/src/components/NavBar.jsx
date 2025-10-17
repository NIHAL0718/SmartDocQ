import React from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

export default function NavBar() {
  const { isAuthenticated, logout } = useAuth()
  const { pathname } = useLocation()
  const navigate = useNavigate()

  const link = (to, label) => (
    <Link className={"nav-link fancy" + (pathname === to ? ' active' : '')} to={to}>
      <span>{label}</span>
    </Link>
  )

  return (
    <nav className="nav glass">
      <div className="brand">SmartDocQ</div>
      <div className="links">
        {isAuthenticated ? (
          <>
            {link('/', 'Home')}
            {link('/upload', 'Upload')}
            {link('/chat', 'Chat')}
            {link('/library', 'Library')}
            {link('/ocr', 'OCR')}
            {link('/translate', 'Translate')}
            <button className="nav-link fancy" onClick={() => { logout(); navigate('/login') }}>
              <span>Logout</span>
            </button>
          </>
        ) : (
          <>
            {link('/login', 'Login')}
            {link('/register', 'Register')}
          </>
        )}
      </div>
    </nav>
  )
}

