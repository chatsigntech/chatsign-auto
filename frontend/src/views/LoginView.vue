<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useAuth } from '../composables/useAuth.js'

const { t } = useI18n()
const router = useRouter()
const { login } = useAuth()

const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  error.value = ''
  loading.value = true
  try {
    await login(username.value, password.value)
    router.push('/')
  } catch {
    error.value = t('login.error')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-card">
      <h1 class="login-title">ChatSign</h1>
      <p class="login-subtitle">Orchestrator</p>

      <n-space vertical :size="16" style="margin-top: 32px;">
        <n-input
          v-model:value="username"
          :placeholder="t('login.username')"
          size="large"
          @keyup.enter="handleLogin"
        />
        <n-input
          v-model:value="password"
          type="password"
          show-password-on="click"
          :placeholder="t('login.password')"
          size="large"
          @keyup.enter="handleLogin"
        />
        <n-alert v-if="error" type="error" :show-icon="false">{{ error }}</n-alert>
        <n-button
          type="primary"
          block
          size="large"
          :loading="loading"
          :disabled="!username || !password"
          @click="handleLogin"
        >
          {{ t('login.submit') }}
        </n-button>
      </n-space>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0A0A0F;
}
.login-card {
  width: 400px;
  padding: 48px 40px;
  background: #1A1A2E;
  border-radius: 12px;
  border: 1px solid rgba(0, 207, 200, 0.15);
}
.login-title {
  font-size: 36px;
  font-weight: 700;
  color: #E2E8F0;
  text-align: center;
  margin-bottom: 0;
}
.login-subtitle {
  text-align: center;
  color: #00CFC8;
  font-size: 16px;
  margin-top: 4px;
}
</style>
