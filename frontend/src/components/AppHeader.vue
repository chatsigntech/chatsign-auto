<script setup>
import { useI18n } from 'vue-i18n'
import { useAuth } from '../composables/useAuth.js'
import { LogOutOutline, SettingsOutline } from '@vicons/ionicons5'

const { t, locale } = useI18n()
const { isAuthenticated, logout } = useAuth()

function toggleLang() {
  locale.value = locale.value === 'zh' ? 'en' : 'zh'
}
</script>

<template>
  <header class="app-header">
    <div class="header-left">
      <router-link to="/" class="header-title">{{ t('app.title') }}</router-link>
      <span class="header-subtitle">{{ t('app.subtitle') }}</span>
    </div>
    <div class="header-right">
      <router-link to="/augmentation" style="text-decoration: none;">
        <n-button quaternary size="small">
          <template #icon><n-icon :component="SettingsOutline" /></template>
          {{ t('augConfig.title') }}
        </n-button>
      </router-link>
      <n-button quaternary size="small" @click="toggleLang">
        {{ locale === 'zh' ? 'EN' : '中文' }}
      </n-button>
      <n-button v-if="isAuthenticated" quaternary size="small" @click="logout">
        <template #icon><n-icon :component="LogOutOutline" /></template>
        {{ t('nav.logout') }}
      </n-button>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 32px;
  border-bottom: 1px solid rgba(0, 207, 200, 0.1);
}
.header-left {
  display: flex;
  align-items: baseline;
  gap: 16px;
}
.header-title {
  font-size: 22px;
  font-weight: 700;
  color: #E2E8F0;
  text-decoration: none;
}
.header-subtitle {
  font-size: 14px;
  color: #00CFC8;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
