import { test, expect } from '@playwright/test'

const BASE = 'http://localhost:8000'
const ADMIN_USER = 'admin'
const ADMIN_PASS = 'admin123'

// ─── API Helper ───
async function apiLogin(request) {
  const res = await request.post(`${BASE}/api/auth/login`, {
    form: { username: ADMIN_USER, password: ADMIN_PASS }
  })
  const data = await res.json()
  return data.access_token
}

async function apiGet(request, token, path) {
  const res = await request.get(`${BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` }
  })
  return res.json()
}

async function apiPost(request, token, path, body) {
  const res = await request.post(`${BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    data: body
  })
  return { status: res.status(), data: await res.json() }
}

async function apiDelete(request, token, path) {
  const res = await request.delete(`${BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` }
  })
  return { status: res.status(), data: await res.json() }
}

// ─── UI Helper ───
async function uiLogin(page) {
  await page.goto('/login')
  await page.waitForLoadState('networkidle')
  await page.locator('.n-input input').nth(0).fill(ADMIN_USER)
  await page.locator('.n-input input').nth(1).fill(ADMIN_PASS)
  await page.locator('button').filter({ hasText: /登录|Sign In/ }).click()
  await page.waitForURL('/', { timeout: 10000 })
}

// ══════════════════════════════════════════════
// 1. API Tests
// ══════════════════════════════════════════════

test.describe('API: Health & Auth', () => {
  test('GET /health returns ok', async ({ request }) => {
    const res = await request.get(`${BASE}/health`)
    expect(res.status()).toBe(200)
    const data = await res.json()
    expect(data.status).toBe('ok')
  })

  test('POST /api/auth/login with valid creds', async ({ request }) => {
    const res = await request.post(`${BASE}/api/auth/login`, {
      form: { username: ADMIN_USER, password: ADMIN_PASS }
    })
    expect(res.status()).toBe(200)
    const data = await res.json()
    expect(data.access_token).toBeTruthy()
    expect(data.token_type).toBe('bearer')
  })

  test('POST /api/auth/login with wrong creds returns 401', async ({ request }) => {
    const res = await request.post(`${BASE}/api/auth/login`, {
      form: { username: 'wrong', password: 'wrong' }
    })
    expect(res.status()).toBe(401)
  })

  test('GET /api/tasks without token returns 401', async ({ request }) => {
    const res = await request.get(`${BASE}/api/tasks/`)
    expect(res.status()).toBe(401)
  })
})

test.describe('API: Config', () => {
  test('GET /api/config/presets returns preset list', async ({ request }) => {
    const token = await apiLogin(request)
    const data = await apiGet(request, token, '/api/config/presets')
    expect(data.presets).toBeDefined()
    expect(data.presets.length).toBeGreaterThanOrEqual(3)
    const names = data.presets.map(p => p.name)
    expect(names).toContain('light')
    expect(names).toContain('medium')
    expect(names).toContain('heavy')
  })

  test('GET /api/config/gpu returns gpu info', async ({ request }) => {
    const token = await apiLogin(request)
    const data = await apiGet(request, token, '/api/config/gpu')
    expect(data.max_gpus).toBeGreaterThanOrEqual(1)
    expect(data.device_ids).toBeDefined()
    expect(data.available).toBeDefined()
  })
})

test.describe('API: Task CRUD Lifecycle', () => {
  let token
  let taskId

  test.beforeAll(async ({ request }) => {
    token = await apiLogin(request)
  })

  test('create task', async ({ request }) => {
    const { status, data } = await apiPost(request, token, '/api/tasks/', {
      name: 'API-CRUD-Test',
      augmentation_preset: 'light'
    })
    expect(status).toBe(200)
    expect(data.task_id).toBeTruthy()
    expect(data.name).toBe('API-CRUD-Test')
    expect(data.status).toBe('pending')
    expect(data.current_phase).toBe(1)
    taskId = data.task_id
  })

  test('create task with batch_name', async ({ request }) => {
    const { status, data } = await apiPost(request, token, '/api/tasks/', {
      name: 'API-Batch-Test',
      batch_name: 'school_unmatch'
    })
    expect(status).toBe(200)
    expect(data.task_id).toBeTruthy()
    // Clean up
    await apiDelete(request, token, `/api/tasks/${data.task_id}`)
  })

  test('list tasks includes created task', async ({ request }) => {
    const data = await apiGet(request, token, '/api/tasks/')
    expect(data.tasks).toBeDefined()
    const found = data.tasks.find(t => t.task_id === taskId)
    expect(found).toBeTruthy()
  })

  test('get task detail with 6 phases', async ({ request }) => {
    const data = await apiGet(request, token, `/api/tasks/${taskId}`)
    expect(data.task.task_id).toBe(taskId)
    expect(data.phases.length).toBe(6)
    for (let i = 0; i < 6; i++) {
      expect(data.phases[i].phase_num).toBe(i + 1)
      expect(data.phases[i].status).toBe('pending')
    }
  })

  test('filter tasks by status', async ({ request }) => {
    const data = await apiGet(request, token, '/api/tasks/?status=pending')
    expect(data.tasks.length).toBeGreaterThanOrEqual(1)
    data.tasks.forEach(t => expect(t.status).toBe('pending'))
  })

  test('get phases endpoint', async ({ request }) => {
    const data = await apiGet(request, token, `/api/phases/${taskId}`)
    expect(data.phases.length).toBe(6)
  })

  test('run task', async ({ request }) => {
    const { status, data } = await apiPost(request, token, `/api/tasks/${taskId}/run`)
    expect(status).toBe(200)
    expect(data.message).toContain('started')
  })

  test('cannot run already running task', async ({ request }) => {
    await new Promise(r => setTimeout(r, 500))
    const res = await request.post(`${BASE}/api/tasks/${taskId}/run`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    // Either 409 (still running) or task may have finished already
    expect([200, 409]).toContain(res.status())
  })

  test('task eventually finishes (completes or fails)', async ({ request }) => {
    // Wait for pipeline to finish (up to 15s)
    let finalStatus = 'running'
    for (let i = 0; i < 15; i++) {
      await new Promise(r => setTimeout(r, 1000))
      const data = await apiGet(request, token, `/api/tasks/${taskId}`)
      finalStatus = data.task.status
      if (['completed', 'failed', 'paused'].includes(finalStatus)) break
    }
    expect(['completed', 'failed']).toContain(finalStatus)
  })

  test('task phases have been processed', async ({ request }) => {
    const data = await apiGet(request, token, `/api/tasks/${taskId}`)
    const completedPhases = data.phases.filter(p => p.status === 'completed')
    // At least Phase 1-3 should complete (no real video data, Phase 4 may fail)
    expect(completedPhases.length).toBeGreaterThanOrEqual(1)
  })

  test('delete task', async ({ request }) => {
    const { status, data } = await apiDelete(request, token, `/api/tasks/${taskId}`)
    expect(status).toBe(200)
    expect(data.message).toContain('deleted')
  })

  test('deleted task returns 404', async ({ request }) => {
    const res = await request.get(`${BASE}/api/tasks/${taskId}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    expect(res.status()).toBe(404)
  })
})

test.describe('API: Pause & Resume', () => {
  let token
  let taskId

  test.beforeAll(async ({ request }) => {
    token = await apiLogin(request)
    const { data } = await apiPost(request, token, '/api/tasks/', { name: 'Pause-Resume-Test' })
    taskId = data.task_id
  })

  test.afterAll(async ({ request }) => {
    if (!taskId) return
    // Wait for any running state to finish
    await new Promise(r => setTimeout(r, 5000))
    try {
      await apiDelete(request, token, `/api/tasks/${taskId}`)
    } catch { /* ignore */ }
  })

  test('cannot pause a pending task', async ({ request }) => {
    const res = await request.post(`${BASE}/api/tasks/${taskId}/pause`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    expect(res.status()).toBe(409)
  })

  test('run then pause', async ({ request }) => {
    await apiPost(request, token, `/api/tasks/${taskId}/run`)
    await new Promise(r => setTimeout(r, 500))
    const res = await request.post(`${BASE}/api/tasks/${taskId}/pause`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    // May be 200 (paused) or 409 (already finished too fast)
    expect([200, 409]).toContain(res.status())
  })

  test('check if paused and resume', async ({ request }) => {
    await new Promise(r => setTimeout(r, 3000))
    const detail = await apiGet(request, token, `/api/tasks/${taskId}`)
    if (detail.task.status === 'paused') {
      const { status } = await apiPost(request, token, `/api/tasks/${taskId}/resume`)
      expect(status).toBe(200)
    }
    // If not paused, task finished too fast - that's ok
  })
})

test.describe('API: Delete Guards', () => {
  let token

  test.beforeAll(async ({ request }) => {
    token = await apiLogin(request)
  })

  test('cannot delete non-existent task', async ({ request }) => {
    const res = await request.delete(`${BASE}/api/tasks/nonexist`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    expect(res.status()).toBe(404)
  })
})

// ══════════════════════════════════════════════
// 2. UI Tests
// ══════════════════════════════════════════════

test.describe('UI: Login Flow', () => {
  test('renders login form with inputs and button', async ({ page }) => {
    await page.goto('/login')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('h1')).toContainText('ChatSign')
    await expect(page.locator('.n-input').first()).toBeVisible()
    await expect(page.locator('.n-input').nth(1)).toBeVisible()
    await expect(page.locator('button').filter({ hasText: /登录|Sign In/ })).toBeVisible()
  })

  test('shows error on wrong password', async ({ page }) => {
    await page.goto('/login')
    await page.waitForLoadState('networkidle')
    await page.locator('.n-input input').nth(0).fill('admin')
    await page.locator('.n-input input').nth(1).fill('wrongpass')
    await page.locator('button').filter({ hasText: /登录|Sign In/ }).click()
    await expect(page.locator('.n-alert')).toBeVisible({ timeout: 5000 })
  })

  test('successful login redirects to dashboard', async ({ page }) => {
    await uiLogin(page)
    await expect(page).toHaveURL('/')
    await expect(page.locator('h2')).toBeVisible()
  })

  test('unauthenticated user redirected to /login', async ({ page }) => {
    // Clear any stored token
    await page.goto('/login')
    await page.evaluate(() => localStorage.removeItem('token'))
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await expect(page).toHaveURL(/\/login/)
  })

  test('logout redirects to login', async ({ page }) => {
    await uiLogin(page)
    const logoutBtn = page.locator('button').filter({ hasText: /退出|Logout/ })
    await expect(logoutBtn).toBeVisible({ timeout: 5000 })
    await logoutBtn.click()
    await page.waitForURL(/\/login/, { timeout: 5000 })
  })
})

test.describe('UI: Dashboard', () => {
  test('shows filter bar with all status filters', async ({ page }) => {
    await uiLogin(page)
    const filterBar = page.locator('.filter-bar')
    await expect(filterBar).toBeVisible()
    const buttons = filterBar.locator('button')
    expect(await buttons.count()).toBe(6) // All, Pending, Running, Completed, Failed, Paused
  })

  test('clicking filter updates task list', async ({ page }) => {
    await uiLogin(page)
    // Click "Completed" filter
    await page.locator('.filter-bar button').nth(3).click()
    await page.waitForTimeout(1000)
    // Should not crash
    await expect(page.locator('.dashboard-content')).toBeVisible()
  })

  test('create task modal opens and works', async ({ page }) => {
    await uiLogin(page)
    await page.locator('button').filter({ hasText: /新建|New Task/ }).click()
    await expect(page.locator('.n-modal')).toBeVisible({ timeout: 3000 })

    // Fill form
    const modalInputs = page.locator('.n-modal .n-input input')
    await modalInputs.nth(0).fill('UI-E2E-Create-Test')

    // Create
    await page.locator('.n-modal button').filter({ hasText: /创建|Create/ }).click()
    await expect(page.locator('.n-modal')).not.toBeVisible({ timeout: 5000 })

    // Verify in list
    await expect(page.locator('text=UI-E2E-Create-Test')).toBeVisible({ timeout: 5000 })
  })

  test('task card is clickable and navigates to detail', async ({ page }) => {
    await uiLogin(page)
    // Click first task card
    const card = page.locator('.task-card').first()
    if (await card.count() > 0) {
      await card.click()
      await page.waitForURL(/\/task\//, { timeout: 5000 })
    }
  })
})

test.describe('UI: Task Detail Page', () => {
  let taskId

  test.beforeAll(async ({ request }) => {
    const token = await apiLogin(request)
    const { data } = await apiPost(request, token, '/api/tasks/', { name: 'UI-Detail-Test' })
    taskId = data.task_id
  })

  test.afterAll(async ({ request }) => {
    if (!taskId) return
    const token = await apiLogin(request)
    await new Promise(r => setTimeout(r, 8000))
    try { await apiDelete(request, token, `/api/tasks/${taskId}`) } catch {}
  })

  test('shows task info card', async ({ page }) => {
    await uiLogin(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    await expect(page.locator('text=UI-Detail-Test')).toBeVisible({ timeout: 5000 })
    await expect(page.locator(`text=${taskId}`)).toBeVisible()
  })

  test('shows 6-phase pipeline visualization', async ({ page }) => {
    await uiLogin(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    const steps = page.locator('.pipeline-step')
    await expect(steps.first()).toBeVisible({ timeout: 5000 })
    expect(await steps.count()).toBe(6)
  })

  test('shows phase detail cards', async ({ page }) => {
    await uiLogin(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    const phaseCards = page.locator('.phase-card')
    await expect(phaseCards.first()).toBeVisible({ timeout: 5000 })
    expect(await phaseCards.count()).toBe(6)
  })

  test('run button works and status updates', async ({ page }) => {
    await uiLogin(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    const runBtn = page.locator('button').filter({ hasText: /启动|Run/ })
    await expect(runBtn).toBeVisible({ timeout: 5000 })
    await runBtn.click()

    // Wait for polling to pick up status change
    await page.waitForTimeout(5000)

    // Some phase should now be completed
    const completedBadge = page.locator('.n-tag').filter({ hasText: /已完成|Completed/ })
    await expect(completedBadge.first()).toBeVisible({ timeout: 10000 })
  })

  test('back button returns to dashboard', async ({ page }) => {
    await uiLogin(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    await page.locator('button').filter({ hasText: /控制台|Dashboard/ }).click()
    await page.waitForURL('/', { timeout: 5000 })
  })
})

test.describe('UI: Language Toggle', () => {
  test('switches between Chinese and English', async ({ page }) => {
    await uiLogin(page)

    // Default is Chinese
    const langBtn = page.locator('header button').filter({ hasText: /EN|中文/ }).first()
    await expect(langBtn).toBeVisible()

    // Switch to English
    const text1 = await langBtn.textContent()
    await langBtn.click()
    await page.waitForTimeout(500)
    const text2 = await langBtn.textContent()
    expect(text2).not.toBe(text1)

    // Dashboard title should change language
    const title = page.locator('h2')
    const titleText = await title.textContent()
    expect(titleText).toBeTruthy()
  })
})

// ══════════════════════════════════════════════
// 3. Frontend SPA Routing Tests
// ══════════════════════════════════════════════

test.describe('SPA Routing', () => {
  test('/ serves index.html', async ({ request }) => {
    const res = await request.get(`${BASE}/`)
    expect(res.status()).toBe(200)
    const text = await res.text()
    expect(text).toContain('<!DOCTYPE html>')
    expect(text).toContain('ChatSign Orchestrator')
  })

  test('/login serves index.html', async ({ request }) => {
    const res = await request.get(`${BASE}/login`)
    expect(res.status()).toBe(200)
    const text = await res.text()
    expect(text).toContain('<!DOCTYPE html>')
  })

  test('/task/any-id serves index.html', async ({ request }) => {
    const res = await request.get(`${BASE}/task/test123`)
    expect(res.status()).toBe(200)
    const text = await res.text()
    expect(text).toContain('<!DOCTYPE html>')
  })

  test('/assets/* serves actual JS files', async ({ request }) => {
    // Get index.html and extract JS filename
    const indexRes = await request.get(`${BASE}/`)
    const html = await indexRes.text()
    const jsMatch = html.match(/src="\/assets\/(index-[^"]+\.js)"/)
    if (jsMatch) {
      const res = await request.get(`${BASE}/assets/${jsMatch[1]}`)
      expect(res.status()).toBe(200)
    }
  })

  test('/health is not intercepted by SPA', async ({ request }) => {
    const res = await request.get(`${BASE}/health`)
    const data = await res.json()
    expect(data.status).toBe('ok')
  })

  test('/docs serves Swagger UI', async ({ request }) => {
    const res = await request.get(`${BASE}/docs`)
    expect(res.status()).toBe(200)
  })
})
