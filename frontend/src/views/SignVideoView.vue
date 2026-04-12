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
let pollTimer = null

// Detail modal
const showDetail = ref(false)
const detailLoading = ref(false)
const detailJob = ref(null)

// Gloss video preview
const glossVideoUrl = ref(null)
const showGlossVideo = ref(false)
const glossVideoTitle = ref('')

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

async function openDetail(job) {
  showDetail.value = true
  detailLoading.value = true
  detailJob.value = null
  try {
    const data = await get(`/api/sign-video/${job.job_id}`)
    detailJob.value = data
  } catch {
    // handled by useApi
  } finally {
    detailLoading.value = false
  }
}

function downloadVideo(job) {
  const a = document.createElement('a')
  a.href = `/api/sign-video/${job.job_id}/video`
  a.download = `${job.title}.mp4`
  a.click()
}

function playGlossVideo(match) {
  if (!match.video_path) return
  // Serve gloss video via the task's phase 3 video endpoint
  // video_path is absolute, we need to stream it — use a generic file endpoint
  // For now, construct a URL from the path
  const parts = match.video_path.split('/')
  const sharedIdx = parts.indexOf('shared')
  if (sharedIdx < 0) return
  const taskId = parts[sharedIdx + 1]
  const filename = parts[parts.length - 1]
  glossVideoUrl.value = `/api/tasks/${taskId}/phases/3/video/${filename}`
  glossVideoTitle.value = match.matched_to || match.gloss
  showGlossVideo.value = true
}

function matchTypeLabel(type) {
  const map = { exact: 'Exact', lemma: 'Lemma', semantic: 'Semantic', none: '-' }
  return map[type] || type
}

function matchTypeColor(type) {
  const map = { exact: 'success', lemma: 'info', semantic: 'warning', none: 'error' }
  return map[type] || 'default'
}

async function deleteJob(job) {
  try {
    await del(`/api/sign-video/${job.job_id}`)
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
          <n-card v-for="job in filteredJobs" :key="job.job_id" class="job-card" size="small"
            hoverable @click="openDetail(job)" style="cursor: pointer;">
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

            <div v-if="job.error_message" class="job-error">{{ job.error_message }}</div>

            <div v-if="isGenerating(job.status)" class="job-progress" @click.stop>
              <n-progress type="line" :percentage="
                job.status === 'extracting' ? 20 :
                job.status === 'matching' ? 50 :
                job.status === 'concatenating' ? 80 : 10
              " :show-indicator="false" status="warning" />
            </div>
          </n-card>
        </div>
        <n-empty v-else-if="!loading" :description="t('signVideo.empty')" style="margin-top: 80px;" />
      </n-spin>
    </div>

    <SignVideoCreateModal v-model:show="showCreate" @created="fetchJobs" />

    <!-- Detail Modal -->
    <n-modal v-model:show="showDetail" preset="card"
      :title="detailJob ? detailJob.title : '...'"
      style="max-width: 720px; max-height: 90vh;" :content-style="{ overflow: 'auto' }">
      <n-spin :show="detailLoading">
        <template v-if="detailJob">
          <!-- Video Player -->
          <div v-if="detailJob.status === 'completed'" class="detail-section">
            <video :src="`/api/sign-video/${detailJob.job_id}/video`" controls
              style="width: 100%; border-radius: 6px; background: #000;" />
            <div style="display: flex; gap: 8px; margin-top: 8px;">
              <n-button size="small" @click="downloadVideo(detailJob)">
                <template #icon><n-icon :component="DownloadOutline" /></template>
                {{ t('signVideo.download') }}
              </n-button>
              <n-popconfirm :positive-text="t('signVideo.delete')" :negative-text="t('signVideo.cancel')"
                @positive-click="deleteJob(detailJob)">
                <template #trigger>
                  <n-button size="small" type="error" ghost>
                    <template #icon><n-icon :component="TrashOutline" /></template>
                    {{ t('signVideo.delete') }}
                  </n-button>
                </template>
                {{ t('signVideo.confirmDelete') }}
              </n-popconfirm>
            </div>
          </div>

          <!-- ASL Gloss Order -->
          <div v-if="detailJob.glosses && detailJob.glosses.length" class="detail-section">
            <div class="detail-label">ASL Gloss</div>
            <div class="asl-glosses">
              <span v-for="g in detailJob.glosses" :key="g" class="asl-gloss-tag">{{ g }}</span>
            </div>
          </div>

          <!-- Match Details -->
          <div v-if="detailJob.match_result && detailJob.match_result.length" class="detail-section">
            <div class="detail-label">Gloss Match</div>
            <div class="match-list">
              <div v-for="m in detailJob.match_result" :key="m.gloss" class="match-row"
                :class="{ clickable: m.video_path }" @click="m.video_path && playGlossVideo(m)">
                <span class="match-gloss">{{ m.gloss }}</span>
                <n-tag :type="matchTypeColor(m.match_type)" size="tiny" round>
                  {{ matchTypeLabel(m.match_type) }}
                </n-tag>
                <span v-if="m.matched_to && m.matched_to !== m.gloss" class="match-to">
                  &rarr; {{ m.matched_to }}
                </span>
                <span class="match-conf">{{ m.confidence > 0 ? (m.confidence * 100).toFixed(0) + '%' : '' }}</span>
                <n-icon v-if="m.video_path" :component="PlayOutline" :size="14" style="opacity: 0.5;" />
              </div>
            </div>
          </div>

          <!-- Original Text -->
          <div class="detail-section">
            <div class="detail-label">{{ t('signVideo.textLabel') }}</div>
            <div class="detail-text">{{ detailJob.input_text }}</div>
          </div>

          <!-- Error -->
          <div v-if="detailJob.error_message" class="detail-section">
            <div class="job-error">{{ detailJob.error_message }}</div>
          </div>
        </template>
      </n-spin>
    </n-modal>

    <!-- Gloss Video Preview -->
    <n-modal v-model:show="showGlossVideo" preset="card" :title="glossVideoTitle"
      style="max-width: 480px;">
      <video v-if="glossVideoUrl" :src="glossVideoUrl" controls autoplay
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
  flex-wrap: wrap;
}
.job-error {
  font-size: 13px;
  color: #e88080;
  margin-top: 6px;
}
.job-progress {
  margin-top: 8px;
}

/* Detail modal */
.detail-section {
  margin-bottom: 20px;
}
.detail-label {
  font-size: 13px;
  font-weight: 600;
  opacity: 0.6;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.asl-glosses {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.asl-gloss-tag {
  background: rgba(0, 207, 200, 0.15);
  color: #00CFC8;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 1px;
}
.match-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.match-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 13px;
}
.match-row.clickable {
  cursor: pointer;
}
.match-row.clickable:hover {
  background: rgba(255, 255, 255, 0.05);
}
.match-gloss {
  font-weight: 600;
  min-width: 100px;
}
.match-to {
  opacity: 0.5;
  font-size: 12px;
}
.match-conf {
  opacity: 0.5;
  font-size: 12px;
  margin-left: auto;
}
.detail-text {
  font-size: 14px;
  line-height: 1.6;
  opacity: 0.8;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
