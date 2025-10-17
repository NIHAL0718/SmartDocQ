import axios from 'axios'

const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api')

const client = axios.create({ baseURL: API_BASE, timeout: 300000 })

function unwrap(resp) { return resp.data }

let bearerToken = null

client.interceptors.request.use((config) => {
  if (bearerToken) config.headers.Authorization = `Bearer ${bearerToken}`
  return config
})

const api = {
  setToken: (t) => { bearerToken = t },
  clearToken: () => { bearerToken = null },
  get: async (path, config) => unwrap(await client.get(path, config)),
  post: async (path, data, config) => unwrap(await client.post(path, data, config)),
}

export default api







