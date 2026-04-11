<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import AppHeader from '../components/AppHeader.vue'
import TaskCard from '../components/TaskCard.vue'
import TaskCreateModal from '../components/TaskCreateModal.vue'
import { AddCircleOutline, VideocamOutline, SettingsOutline, LanguageOutline } from '@vicons/ionicons5'

const { t } = useI18n()
const { get } = useApi()

const tasks = ref([])
const loading = ref(false)
const filter = ref(null)
const showCreate = ref(false)
const accuracyUrl = ref('')
let pollTimer = null

const filters = [
  { key: null, label: 'dashboard.filterAll' },
  { key: 'pending', label: 'dashboard.filterPending' },
  { key: 'running', label: 'dashboard.filterRunning' },
  { key: 'completed', label: 'dashboard.filterCompleted' },
  { key: 'failed', label: 'dashboard.filterFailed' },
  { key: 'paused', label: 'dashboard.filterPaused' }
]

async function fetchTasks() {
  loading.value = true
  try {
    const url = filter.value ? `/api/tasks/?status=${filter.value}` : '/api/tasks/'
    const data = await get(url)
    tasks.value = data.tasks
  } catch {
    // handled by useApi
  } finally {
    loading.value = false
  }
}

function onFilterChange(key) {
  filter.value = key
  fetchTasks()
}

function onTaskCreated() {
  fetchTasks()
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(fetchTasks, 5000)
}

function handleVisibility() {
  if (document.hidden) {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
  } else {
    fetchTasks()
    startPoll()
  }
}

onMounted(async () => {
  fetchTasks()
  startPoll()
  document.addEventListener('visibilitychange', handleVisibility)
  try {
    const data = await get('/api/config/accuracy-url')
    accuracyUrl.value = data.url || ''
  } catch {}
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  document.removeEventListener('visibilitychange', handleVisibility)
})
</script>

<template>
  <div class="dashboard-page">
    <AppHeader />

    <div class="dashboard-content">
      <div class="dashboard-toolbar">
        <h2 class="page-title">{{ t('dashboard.title') }}</h2>
        <n-space>
          <n-button @click="$router.push('/sign-video')">
            <template #icon><n-icon :component="LanguageOutline" /></template>
            {{ t('signVideo.title') }}
          </n-button>
          <n-button v-if="accuracyUrl" tag="a" :href="accuracyUrl" target="_blank">
            <template #icon><n-icon :component="VideocamOutline" /></template>
            {{ t('accuracy.title') }}
          </n-button>
          <n-button @click="$router.push('/augmentation')">
            <template #icon><n-icon :component="SettingsOutline" /></template>
            {{ t('augConfig.title') }}
          </n-button>
          <n-button type="primary" @click="showCreate = true">
            <template #icon><n-icon :component="AddCircleOutline" /></template>
            {{ t('dashboard.create') }}
          </n-button>
        </n-space>
      </div>

      <div class="filter-bar">
        <n-button
          v-for="f in filters"
          :key="f.key"
          :type="filter === f.key ? 'primary' : 'default'"
          :ghost="filter !== f.key"
          size="small"
          @click="onFilterChange(f.key)"
        >
          {{ t(f.label) }}
        </n-button>
      </div>

      <n-spin :show="loading && tasks.length === 0">
        <div v-if="tasks.length" class="task-grid">
          <TaskCard v-for="task in tasks" :key="task.task_id" :task="task" />
        </div>
        <n-empty v-else-if="!loading" :description="t('dashboard.empty')" style="margin-top: 80px;" />
      </n-spin>
    </div>

    <TaskCreateModal v-model:show="showCreate" @created="onTaskCreated" />
  </div>
</template>

<style scoped>
.dashboard-page {
  min-height: 100vh;
}
.dashboard-content {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px 24px;
}
.dashboard-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}
.page-title {
  font-size: 22px;
  font-weight: 600;
}
.filter-bar {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 16px;
}
</style>
