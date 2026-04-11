<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import { formatDate } from '../utils/format.js'
import AppHeader from '../components/AppHeader.vue'
import SignVideoCreateModal from '../components/SignVideoCreateModal.vue'
import { AddCircleOutline, PlayOutline, DownloadOutline, TrashOutline, ArrowBackOutline } from '@vicons/ionicons5'

const { t } = useI18n()
const { get, del } = useApi()

const jobs = ref([])
const loading = ref(false)
const filter = ref(null)
const showCreate = ref(false)
const previewUrl = ref(null)
const showPreview = ref(false)
let pollTimer = null

const filters = [
  { key: null, label: 'signVideo.filterAll' },
  { key: 'generating', label: 'signVideo.filterGenerating' },
  { key: 'completed', label: 'signVideo.filterCompleted' },
  { key: 'failed', label: 'signVideo.filterFailed' },
]

const generatingStatuses = ['pending', 'extracting', 'matching', 'concatenating']

const filteredJobs = computed(() => {
  if (!filter.value) return jobs.value
  if (filter.value === 'generating') {
    return jobs.value.filter(j => generatingStatuses.includes(j.status))
  }
  return jobs.value.filter(j => j.status === filter.value)
})

async function fetchJobs() {
  loading.value = true
  try {
    const data = await get('/api/sign-video/')
    jobs.value = data.jobs
  } catch {
    // handled by useApi
  } finally {
    loading.value = false
  }
}

function onFilterChange(key) {
  filter.value = key
}

function statusText(status) {
  const map = {
    pending: t('signVideo.statusPending'),
    extracting: t('signVideo.statusExtracting'),
    matching: t('signVideo.statusMatching'),
    concatenating: t('signVideo.statusConcatenating'),
    completed: t('status.completed'),
    failed: t('status.failed'),
  }
  return map[status] || status
}

function statusType(status) {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'error'
  return 'warning'
}

function isGenerating(status) {
  return generatingStatuses.includes(status)
}

function formatDuration(sec) {
  if (!sec) return '-'
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function openPreview(job) {
  previewUrl.value = `/api/sign-video/${job.job_id}/video`
  showPreview.value = true
}

function downloadVideo(job) {
  const a = document.createElement('a')
  a.href = `/api/sign-video/${job.job_id}/video`
  a.download = `${job.title}.mp4`
  a.click()
}

async function deleteJob(job) {
  try {
    await del(`/api/sign-video/${job.job_id}`)
    fetchJobs()
  } catch {
    // handled by useApi
  }
}

function startPoll() {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(fetchJobs, 5000)
}

function handleVisibility() {
  if (document.hidden) {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
  } else {
    fetchJobs()
    startPoll()
  }
}

onMounted(() => {
  fetchJobs()
  startPoll()
  document.addEventListener('visibilitychange', handleVisibility)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  document.removeEventListener('visibilitychange', handleVisibility)
})
</script>

<template>
  <div class="sign-video-page">
    <AppHeader />

    <div class="page-content">
      <div class="page-toolbar">
        <div style="display: flex; align-items: center; gap: 12px;">
          <n-button text @click="$router.push('/')">
            <template #icon><n-icon :component="ArrowBackOutline" /></template>
          </n-button>
          <h2 class="page-title">{{ t('signVideo.title') }}</h2>
        </div>
        <n-button type="primary" @click="showCreate = true">
          <template #icon><n-icon :component="AddCircleOutline" /></template>
          {{ t('signVideo.create') }}
        </n-button>
      </div>

      <div class="filter-bar">
        <n-button
          v-for="f in filters" :key="f.key"
          :type="filter === f.key ? 'primary' : 'default'"
          :ghost="filter !== f.key"
          size="small"
          @click="onFilterChange(f.key)"
        >
          {{ t(f.label) }}
        </n-button>
      </div>

      <n-spin :show="loading && jobs.length === 0">
        <div v-if="filteredJobs.length" class="job-list">
          <n-card v-for="job in filteredJobs" :key="job.job_id" class="job-card" size="small">
            <div class="job-header">
              <span class="job-title">{{ job.title }}</span>
              <n-tag :type="statusType(job.status)" size="small" round>
                {{ statusText(job.status) }}
              </n-tag>
            </div>

            <div class="job-meta">
              <span>{{ formatDate(job.created_at) }}</span>
              <span v-if="job.gloss_count">{{ t('signVideo.glossCount') }}: {{ job.gloss_count }}</span>
              <span v-if="job.matched_count">{{ t('signVideo.matchedCount') }}: {{ job.matched_count }}</span>
              <span v-if="job.duration_sec">{{ t('signVideo.duration') }}: {{ formatDuration(job.duration_sec) }}</span>
            </div>

            <div class="job-text">{{ job.input_text }}</div>

            <div v-if="job.error_message" class="job-error">{{ job.error_message }}</div>

            <div v-if="isGenerating(job.status)" class="job-progress">
              <n-progress type="line" :percentage="
                job.status === 'extracting' ? 20 :
                job.status === 'matching' ? 50 :
                job.status === 'concatenating' ? 80 : 10
              " :show-indicator="false" status="warning" />
            </div>

            <div class="job-actions" v-if="job.status === 'completed' || job.status === 'failed'">
              <n-button v-if="job.status === 'completed'" size="small" @click="openPreview(job)">
                <template #icon><n-icon :component="PlayOutline" /></template>
                {{ t('signVideo.preview') }}
              </n-button>
              <n-button v-if="job.status === 'completed'" size="small" @click="downloadVideo(job)">
                <template #icon><n-icon :component="DownloadOutline" /></template>
                {{ t('signVideo.download') }}
              </n-button>
              <n-popconfirm :positive-text="t('signVideo.delete')" :negative-text="t('signVideo.cancel')"
                @positive-click="deleteJob(job)">
                <template #trigger>
                  <n-button size="small" type="error" ghost>
                    <template #icon><n-icon :component="TrashOutline" /></template>
                    {{ t('signVideo.delete') }}
                  </n-button>
                </template>
                {{ t('signVideo.confirmDelete') }}
              </n-popconfirm>
            </div>
          </n-card>
        </div>
        <n-empty v-else-if="!loading" :description="t('signVideo.empty')" style="margin-top: 80px;" />
      </n-spin>
    </div>

    <SignVideoCreateModal v-model:show="showCreate" @created="fetchJobs" />

    <n-modal v-model:show="showPreview" preset="card" :title="t('signVideo.preview')"
      style="max-width: 580px;">
      <video v-if="previewUrl" :src="previewUrl" controls autoplay
        style="width: 100%; border-radius: 6px; background: #000;" />
    </n-modal>
  </div>
</template>

<style scoped>
.sign-video-page {
  min-height: 100vh;
}
.page-content {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px;
}
.page-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}
.page-title {
  font-size: 22px;
  font-weight: 600;
  margin: 0;
}
.filter-bar {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.job-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.job-card {
  border-radius: 8px;
}
.job-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.job-title {
  font-size: 16px;
  font-weight: 600;
}
.job-meta {
  display: flex;
  gap: 16px;
  font-size: 13px;
  opacity: 0.7;
  margin-bottom: 6px;
  flex-wrap: wrap;
}
.job-text {
  font-size: 13px;
  opacity: 0.6;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 8px;
}
.job-error {
  font-size: 13px;
  color: #e88080;
  margin-bottom: 8px;
}
.job-progress {
  margin-bottom: 8px;
}
.job-actions {
  display: flex;
  gap: 8px;
}
</style>
