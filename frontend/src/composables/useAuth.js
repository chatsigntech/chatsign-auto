import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

const token = ref(localStorage.getItem('token') || '')

export function useAuth() {
  const router = useRouter()
  const isAuthenticated = computed(() => !!token.value)

  async function login(username, password) {
    const body = new URLSearchParams({ username, password })
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      body
    })
    if (!res.ok) throw new Error('Login failed')
    const data = await res.json()
    token.value = data.access_token
    localStorage.setItem('token', data.access_token)
  }

  function logout() {
    token.value = ''
    localStorage.removeItem('token')
    router.push('/login')
  }

  return { token, isAuthenticated, login, logout }
}
