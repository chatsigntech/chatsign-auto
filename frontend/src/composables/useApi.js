import { useAuth } from './useAuth.js'

export function useApi() {
  const { token, logout } = useAuth()

  async function request(url, options = {}) {
    const headers = {
      'Authorization': `Bearer ${token.value}`,
      ...options.headers
    }
    if (options.body && !(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json'
      options.body = JSON.stringify(options.body)
    }
    const res = await fetch(url, { ...options, headers })
    if (res.status === 401) {
      logout()
      throw new Error('Unauthorized')
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || 'Request failed')
    }
    return res.json()
  }

  const get = (url) => request(url)
  const post = (url, body) => request(url, { method: 'POST', body })
  const put = (url, body) => request(url, { method: 'PUT', body })
  const del = (url) => request(url, { method: 'DELETE' })

  return { get, post, put, del }
}
