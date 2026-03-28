import { test, expect } from '@playwright/test'

const ADMIN_USER = 'admin'
const ADMIN_PASS = 'admin123'

// Helper: login and return page with auth
async function login(page) {
  await page.goto('/login')
  await page.waitForLoadState('networkidle')

  // Check login page renders
  const title = page.locator('h1')
  await expect(title).toBeVisible({ timeout: 10000 })

  // Fill credentials
  const inputs = page.locator('input')
  await inputs.nth(0).fill(ADMIN_USER)
  await inputs.nth(1).fill(ADMIN_PASS)

  // Click login button
  await page.locator('button[type="button"]').filter({ hasText: /登录|Sign In/ }).click()

  // Should redirect to dashboard
  await page.waitForURL('/', { timeout: 10000 })
  return page
}

test.describe('Login Page', () => {
  test('should render login form', async ({ page }) => {
    await page.goto('/login')
    await page.waitForLoadState('networkidle')

    // Check page loads
    await expect(page.locator('h1')).toContainText('ChatSign')
    await expect(page.locator('input').first()).toBeVisible()
  })

  test('should show error on wrong credentials', async ({ page }) => {
    await page.goto('/login')
    await page.waitForLoadState('networkidle')

    const inputs = page.locator('input')
    await inputs.nth(0).fill('wrong')
    await inputs.nth(1).fill('wrong')

    await page.locator('button[type="button"]').filter({ hasText: /登录|Sign In/ }).click()

    // Should show error
    await expect(page.locator('.n-alert')).toBeVisible({ timeout: 5000 })
  })

  test('should login successfully and redirect', async ({ page }) => {
    await login(page)
    // Should be on dashboard
    await expect(page).toHaveURL('/')
  })

  test('should redirect unauthenticated to login', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await expect(page).toHaveURL(/\/login/)
  })
})

test.describe('Dashboard', () => {
  test('should show dashboard after login', async ({ page }) => {
    await login(page)

    // Dashboard title should be visible
    await expect(page.locator('h2')).toBeVisible({ timeout: 5000 })
  })

  test('should show filter buttons', async ({ page }) => {
    await login(page)

    // Filter buttons
    const buttons = page.locator('.filter-bar button')
    await expect(buttons.first()).toBeVisible({ timeout: 5000 })
    expect(await buttons.count()).toBeGreaterThanOrEqual(5)
  })

  test('should open create task modal', async ({ page }) => {
    await login(page)

    // Click create button
    await page.locator('button').filter({ hasText: /新建|New Task/ }).click()

    // Modal should appear with inputs
    await expect(page.locator('.n-modal')).toBeVisible({ timeout: 5000 })
    await expect(page.locator('.n-modal input').first()).toBeVisible()
  })

  test('should create a task and show it in list', async ({ page }) => {
    await login(page)

    // Open modal
    await page.locator('button').filter({ hasText: /新建|New Task/ }).click()
    await expect(page.locator('.n-modal')).toBeVisible({ timeout: 5000 })

    // Fill task name
    const modalInputs = page.locator('.n-modal input')
    await modalInputs.first().fill('Playwright Test Task')

    // Click create
    await page.locator('.n-modal button').filter({ hasText: /创建|Create/ }).click()

    // Modal should close
    await expect(page.locator('.n-modal')).not.toBeVisible({ timeout: 5000 })

    // Task should appear in list
    await expect(page.locator('text=Playwright Test Task')).toBeVisible({ timeout: 5000 })
  })
})

test.describe('Task Detail', () => {
  let taskId

  test.beforeAll(async ({ request }) => {
    // Create a task via API for detail tests
    const loginRes = await request.post('/api/auth/login', {
      form: { username: ADMIN_USER, password: ADMIN_PASS }
    })
    const { access_token } = await loginRes.json()

    const createRes = await request.post('/api/tasks/', {
      headers: { Authorization: `Bearer ${access_token}`, 'Content-Type': 'application/json' },
      data: { name: 'Detail Test Task' }
    })
    const task = await createRes.json()
    taskId = task.task_id
  })

  test.afterAll(async ({ request }) => {
    if (!taskId) return
    const loginRes = await request.post('/api/auth/login', {
      form: { username: ADMIN_USER, password: ADMIN_PASS }
    })
    const { access_token } = await loginRes.json()
    await request.delete(`/api/tasks/${taskId}`, {
      headers: { Authorization: `Bearer ${access_token}` }
    })
  })

  test('should navigate to task detail', async ({ page }) => {
    await login(page)

    // Click on the task card
    await page.locator(`text=${taskId}`).first().click()

    // Should be on detail page
    await page.waitForURL(`/task/${taskId}`, { timeout: 10000 })

    // Task name visible
    await expect(page.locator('text=Detail Test Task')).toBeVisible({ timeout: 5000 })
  })

  test('should show 6 phase pipeline', async ({ page }) => {
    await login(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    // Pipeline steps
    const steps = page.locator('.pipeline-step')
    await expect(steps.first()).toBeVisible({ timeout: 10000 })
    expect(await steps.count()).toBe(6)
  })

  test('should show run button for pending task', async ({ page }) => {
    await login(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    const runBtn = page.locator('button').filter({ hasText: /启动|Run/ })
    await expect(runBtn).toBeVisible({ timeout: 5000 })
  })

  test('should run task and show progress', async ({ page }) => {
    await login(page)
    await page.goto(`/task/${taskId}`)
    await page.waitForLoadState('networkidle')

    // Click run
    await page.locator('button').filter({ hasText: /启动|Run/ }).click()

    // Wait for status to change from pending
    await page.waitForTimeout(3000)

    // Should show completed or failed phases (pipeline runs fast with no data)
    const phaseCards = page.locator('.phase-card')
    await expect(phaseCards.first()).toBeVisible({ timeout: 10000 })
  })

  test('should show delete button and delete task', async ({ page }) => {
    await login(page)

    // Create a fresh task for deletion
    const loginRes = await page.request.post('/api/auth/login', {
      form: { username: ADMIN_USER, password: ADMIN_PASS }
    })
    const { access_token } = await loginRes.json()
    const createRes = await page.request.post('/api/tasks/', {
      headers: { Authorization: `Bearer ${access_token}`, 'Content-Type': 'application/json' },
      data: { name: 'Delete Me Task' }
    })
    const task = await createRes.json()

    await page.goto(`/task/${task.task_id}`)
    await page.waitForLoadState('networkidle')

    // Click delete
    const deleteBtn = page.locator('button').filter({ hasText: /删除|Delete/ })
    await expect(deleteBtn).toBeVisible({ timeout: 5000 })
    await deleteBtn.click()

    // Confirm dialog
    const confirmBtn = page.locator('.n-dialog button').filter({ hasText: /删除|Delete/ })
    await expect(confirmBtn).toBeVisible({ timeout: 3000 })
    await confirmBtn.click()

    // Should redirect to dashboard
    await page.waitForURL('/', { timeout: 10000 })
  })
})

test.describe('Language Toggle', () => {
  test('should switch between Chinese and English', async ({ page }) => {
    await login(page)

    // Find language toggle button
    const langBtn = page.locator('button').filter({ hasText: /EN|中文/ }).first()
    await expect(langBtn).toBeVisible({ timeout: 5000 })

    const initialText = await langBtn.textContent()
    await langBtn.click()
    await page.waitForTimeout(500)

    const newText = await langBtn.textContent()
    expect(newText).not.toBe(initialText)
  })
})
