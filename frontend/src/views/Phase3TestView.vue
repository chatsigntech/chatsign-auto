<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import { formatDate } from '../utils/format.js'
import AppHeader from '../components/AppHeader.vue'
import Phase3VideoSelectorModal from '../components/Phase3VideoSelectorModal.vue'
import { AddCircleOutline, TrashOutline, ArrowBackOutline } from '@vicons/ionicons5'

const { t } = useI18n()
const { get, post, del } = useApi()

const jobs = ref([])
const loading = ref(false)
const filter = ref(null)
const showSelector = ref(false)
const showDetail = ref(false)
const detailJob = ref(null)
let pollTimer = null

const filters = [
  { key: null, label: 'phase3Test.filterAll' },
  { key: 'running', label: 'phase3Test.filterRunning' },
  { key: 'completed', label: 'phase3Test.filterCompleted' },
  { key: 'failed', label: 'phase3Test.filterFailed' },
]

const runningStatuses = ['pending', 'transfer', 'processing', 'framer']

const filteredJobs = computed(() => {
  if (!filter.value) return jobs.value
  if (filter.value === 'running') {
    return jobs.value.filter(j => runningStatuses.includes(j.status))
  }
  return jobs.value.filter(j => j.status === filter.value)
})

async function fetchJobs() {
  loading.value = true
  try {
    const data = await get('/api/phase3-test/jobs')
    jobs.value = data.jobs
  } catch {
    // handled by useApi
  } finally {
    loading.value = false
  }
}

function statusText(status) {
  const map = {
    pending: t('phase3Test.statusPending'),
    transfer: t('phase3Test.statusTransfer'),
    processing: t('phase3Test.statusProcessing'),
    framer: t('phase3Test.statusFramer'),
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

function formatSec(sec) {
  if (!sec) return '-'
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return `${m}m ${s}s`
}

async function startJob(videoId) {
  try {
    await post('/api/phase3-test/run', { video_id: videoId })
    fetchJobs()
  } catch {
    // handled by useApi
  }
}

async function openDetail(job) {
  showDetail.value = true
  detailJob.value = null
  try {
    detailJob.value = await get(`/api/phase3-test/jobs/${job.job_id}`)
  } catch {
    // handled by useApi
  }
}

async function deleteJob(job) {
  try {
    await del(`/api/phase3-test/jobs/${job.job_id}`)
    showDetail.value = false
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
  <div class="p3test-page">
    <AppHeader />

    <div class="page-content">
      <div class="page-toolbar">
        <div style="display: flex; align-items: center; gap: 12px;">
          <n-button text @click="$router.push('/')">
            <template #icon><n-icon :component="ArrowBackOutline" /></template>
          </n-button>
          <h2 class="page-title">{{ t('phase3Test.title') }}</h2>
        </div>
        <n-button type="primary" @click="showSelector = true">
          <template #icon><n-icon :component="AddCircleOutline" /></template>
          {{ t('phase3Test.selectVideo') }}
        </n-button>
      </div>

      <div class="filter-bar">
        <n-button v-for="f in filters" :key="f.key"
          :type="filter === f.key ? 'primary' : 'default'"
          :ghost="filter !== f.key" size="small"
          @click="filter = f.key">
          {{ t(f.label) }}
        </n-button>
      </div>

      <n-spin :show="loading && jobs.length === 0">
        <div v-if="filteredJobs.length" class="job-list">
          <n-card v-for="job in filteredJobs" :key="job.job_id" class="job-card" size="small"
            hoverable style="cursor: pointer;" @click="openDetail(job)">
            <div class="job-header">
              <span class="job-title">{{ job.source_filename }}</span>
              <n-tag :type="statusType(job.status)" size="small" round>
                {{ statusText(job.status) }}
              </n-tag>
            </div>
            <div class="job-meta">
              <span>{{ job.sentence_text }}</span>
            </div>
            <div class="job-meta">
              <span>{{ formatDate(job.created_at) }}</span>
              <span v-if="job.translator_id">{{ job.translator_id }}</span>
              <span v-if="job.duration_sec">{{ t('phase3Test.totalTime') }}: {{ formatSec(job.duration_sec) }}</span>
            </div>
            <div v-if="job.error_message" class="job-error">{{ job.error_message }}</div>
            <div v-if="runningStatuses.includes(job.status)" style="margin-top: 8px;">
              <n-progress type="line" :percentage="
                job.status === 'transfer' ? 25 :
                job.status === 'processing' ? 55 :
                job.status === 'framer' ? 80 : 10
              " :show-indicator="false" status="warning" />
            </div>
          </n-card>
        </div>
        <n-empty v-else-if="!loading" :description="t('phase3Test.noJobs')" style="margin-top: 80px;" />
      </n-spin>
    </div>

    <Phase3VideoSelectorModal v-model:show="showSelector" @selected="startJob" />

    <!-- Detail / Comparison Modal -->
    <n-modal v-model:show="showDetail" preset="card"
      :title="detailJob ? detailJob.source_filename : '...'"
      style="max-width: 900px; max-height: 90vh;" :content-style="{ overflow: 'auto' }">
      <template v-if="detailJob">
        <!-- Side-by-side comparison -->
        <div v-if="detailJob.status === 'completed'" class="comparison">
          <div class="compare-col">
            <div class="compare-label">{{ t('phase3Test.original') }}</div>
            <video :src="`/api/phase3-test/jobs/${detailJob.job_id}/original-video`"
              controls style="width: 100%; border-radius: 6px; background: #000;" />
          </div>
          <div class="compare-col">
            <div class="compare-label">{{ t('phase3Test.generated') }}</div>
            <video :src="`/api/phase3-test/jobs/${detailJob.job_id}/generated-video`"
              controls style="width: 100%; border-radius: 6px; background: #000;" />
          </div>
        </div>

        <!-- Info -->
        <div class="detail-section">
          <div class="detail-label">{{ t('phase3Test.sentence') }}</div>
          <div>{{ detailJob.sentence_text }}</div>
        </div>

        <!-- Timing breakdown -->
        <div v-if="detailJob.duration_sec" class="detail-section">
          <div class="detail-label">Timing</div>
          <div class="timing-grid">
            <span>{{ t('phase3Test.transferTime') }}</span><span>{{ formatSec(detailJob.transfer_time_sec) }}</span>
            <span>{{ t('phase3Test.processTime') }}</span><span>{{ formatSec(detailJob.process_time_sec) }}</span>
            <span>{{ t('phase3Test.framerTime') }}</span><span>{{ formatSec(detailJob.framer_time_sec) }}</span>
            <span style="font-weight: 600;">{{ t('phase3Test.totalTime') }}</span><span style="font-weight: 600;">{{ formatSec(detailJob.duration_sec) }}</span>
          </div>
        </div>

        <!-- Error -->
        <div v-if="detailJob.error_message" class="detail-section">
          <div class="job-error">{{ detailJob.error_message }}</div>
        </div>

        <!-- Delete -->
        <div style="margin-top: 16px;">
          <n-popconfirm :positive-text="t('phase3Test.delete')" :negative-text="t('phase3Test.cancel')"
            @positive-click="deleteJob(detailJob)">
            <template #trigger>
              <n-button type="error" ghost size="small">
                <template #icon><n-icon :component="TrashOutline" /></template>
                {{ t('phase3Test.delete') }}
              </n-button>
            </template>
            {{ t('phase3Test.confirmDelete') }}
          </n-popconfirm>
        </div>
      </template>
    </n-modal>
  </div>
</template>

<style scoped>
.p3test-page { min-height: 100vh; }
.page-content { max-width: 960px; margin: 0 auto; padding: 24px; }
.page-toolbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
.page-title { font-size: 22px; font-weight: 600; margin: 0; }
.filter-bar { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
.job-list { display: flex; flex-direction: column; gap: 12px; }
.job-card { border-radius: 8px; }
.job-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
.job-title { font-size: 15px; font-weight: 600; }
.job-meta { display: flex; gap: 16px; font-size: 13px; opacity: 0.7; flex-wrap: wrap; }
.job-error { font-size: 13px; color: #e88080; margin-top: 4px; }

.comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
.compare-col { text-align: center; }
.compare-label { font-size: 13px; font-weight: 600; opacity: 0.6; margin-bottom: 6px; text-transform: uppercase; }
.detail-section { margin-bottom: 16px; }
.detail-label { font-size: 13px; font-weight: 600; opacity: 0.6; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.timing-grid { display: grid; grid-template-columns: 120px auto; gap: 4px 12px; font-size: 13px; }
</style>
