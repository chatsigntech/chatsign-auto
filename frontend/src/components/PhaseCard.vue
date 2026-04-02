<script setup>
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import StatusBadge from './StatusBadge.vue'
import { formatDate } from '../utils/format.js'

const props = defineProps({ phase: Object, taskId: String })
const { t } = useI18n()
const { get } = useApi()

const summary = ref(null)
const accuracyProgress = ref(null)
const files = ref([])
const selectedFile = ref(null)
const fileContent = ref('')
const showModal = ref(false)
const loadingContent = ref(false)
const expanded = ref(false)

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

async function loadFiles() {
  if (!props.taskId || files.value.length > 0) return
  try {
    const data = await get(`/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/files`)
    files.value = data.files || []
  } catch { /* ignore */ }
}

async function viewFile(file) {
  if (!file.is_text) return
  selectedFile.value = file
  loadingContent.value = true
  showModal.value = true
  try {
    const data = await get(`/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/files/${file.path}`)
    fileContent.value = data.content || ''
  } catch (e) {
    fileContent.value = `Error loading file: ${e}`
  } finally {
    loadingContent.value = false
  }
}

function toggleExpand() {
  expanded.value = !expanded.value
  if (expanded.value) loadFiles()
}

async function loadSummary() {
  if (!props.taskId || summary.value) return
  try {
    const data = await get(`/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/summary`)
    if (data && Object.keys(data).length > 0) summary.value = data
  } catch { /* ignore */ }
}

let accuracyPollTimer = null

async function loadAccuracyProgress() {
  if (!props.taskId || props.phase.phase_num !== 3) return
  try {
    const data = await get(`/api/tasks/${props.taskId}/phases/3/accuracy-progress`)
    if (data && !data.detail) accuracyProgress.value = data
  } catch { /* ignore */ }
}

function startAccuracyPolling() {
  if (accuracyPollTimer) return
  loadAccuracyProgress()
  accuracyPollTimer = setInterval(loadAccuracyProgress, 10000)
}

function stopAccuracyPolling() {
  if (accuracyPollTimer) { clearInterval(accuracyPollTimer); accuracyPollTimer = null }
}

watch(() => props.phase?.status, (s) => {
  if (s === 'completed') {
    loadSummary()
    if (expanded.value) loadFiles()
    stopAccuracyPolling()
  }
  // Phase 3 pending = poll accuracy progress
  if (props.phase.phase_num === 3 && (s === 'pending' || s === 'running')) {
    startAccuracyPolling()
  }
}, { immediate: true })

import { onUnmounted } from 'vue'
onUnmounted(stopAccuracyPolling)
</script>

<template>
  <n-card size="small" class="phase-card">
    <div class="phase-header" @click="toggleExpand" style="cursor: pointer;">
      <span class="phase-title">
        {{ t('task.phase') }} {{ phase.phase_num }} — {{ t(`phases.${phase.phase_num}`) }}
      </span>
      <n-space :size="8" align="center">
        <n-tag v-if="files.length > 0" size="small" :bordered="false" type="info">
          {{ files.length }} files
        </n-tag>
        <StatusBadge :status="phase.status" />
      </n-space>
    </div>

    <n-progress
      v-if="phase.status === 'running' || phase.progress > 0"
      type="line"
      :percentage="Math.round(phase.progress)"
      :height="6"
      color="#00CFC8"
      style="margin: 8px 0;"
    />

    <div class="phase-meta">
      <span v-if="phase.started_at">
        <span class="meta-label">Start</span> {{ formatDate(phase.started_at) }}
      </span>
      <span v-if="phase.completed_at">
        <span class="meta-label">End</span> {{ formatDate(phase.completed_at) }}
      </span>
      <span v-if="phase.gpu_id != null">
        <span class="meta-label">{{ t('task.gpuId') }}</span> {{ phase.gpu_id }}
      </span>
    </div>

    <!-- Phase 3: accuracy link + live progress -->
    <div v-if="phase.phase_num === 3" class="phase-summary" style="margin-top: 8px;">
      <div class="summary-row">
        <span class="summary-key">Recording site</span>
        <a href="https://accuracy.chatsign.ai" target="_blank" class="summary-val summary-link">https://accuracy.chatsign.ai</a>
      </div>
      <template v-if="accuracyProgress">
        <div class="accuracy-progress">
          <div class="progress-item">
            <span class="progress-label">Recording</span>
            <n-progress type="line" :percentage="accuracyProgress.total_glosses ? Math.round(accuracyProgress.recorded / accuracyProgress.total_glosses * 100) : 0" :height="8" color="#00CFC8" />
            <span class="progress-text">{{ accuracyProgress.recorded }} / {{ accuracyProgress.total_glosses }}</span>
          </div>
          <div class="progress-item">
            <span class="progress-label">Review</span>
            <n-progress type="line" :percentage="accuracyProgress.recorded ? Math.round((accuracyProgress.approved + accuracyProgress.rejected) / accuracyProgress.recorded * 100) : 0" :height="8" :color="accuracyProgress.rejected > 0 ? '#f0a020' : '#18a058'" />
            <span class="progress-text">
              <n-tag size="tiny" type="success" :bordered="false">{{ accuracyProgress.approved }} approved</n-tag>
              <n-tag v-if="accuracyProgress.rejected > 0" size="tiny" type="warning" :bordered="false">{{ accuracyProgress.rejected }} rejected</n-tag>
              <n-tag v-if="accuracyProgress.pending_review > 0" size="tiny" :bordered="false">{{ accuracyProgress.pending_review }} pending</n-tag>
            </span>
          </div>
        </div>
      </template>
    </div>

    <!-- Summary -->
    <div v-if="summary" class="phase-summary">
      <div v-for="(val, key) in summary" :key="key" class="summary-row">
        <span class="summary-key">{{ key.replace(/_/g, ' ') }}</span>
        <span v-if="Array.isArray(val)" class="summary-val summary-list">
          <n-tag v-for="item in val" :key="item" size="tiny" :bordered="false" type="info" style="margin: 1px;">{{ item }}</n-tag>
        </span>
        <a v-else-if="typeof val === 'string' && val.startsWith('http')" :href="val" target="_blank" class="summary-val summary-link">{{ val }}</a>
        <span v-else-if="typeof val === 'string' && val.length > 60" class="summary-val summary-long">{{ val }}</span>
        <n-tag v-else size="small" :bordered="false" :type="typeof val === 'number' && val > 0 ? 'success' : 'default'">
          <span class="summary-val">{{ val }}</span>
        </n-tag>
      </div>
    </div>

    <n-alert v-if="phase.error_message" type="error" :title="t('task.errorMessage')" style="margin-top: 8px;">
      {{ phase.error_message }}
    </n-alert>

    <!-- File list -->
    <div v-if="expanded && files.length > 0" class="file-list">
      <div
        v-for="file in files"
        :key="file.path"
        class="file-item"
        :class="{ clickable: file.is_text }"
        @click="viewFile(file)"
      >
        <span class="file-icon">{{ file.is_text ? '📄' : (file.path.endsWith('.mp4') ? '🎬' : '📦') }}</span>
        <span class="file-name">{{ file.path }}</span>
        <span class="file-size">{{ formatSize(file.size) }}</span>
      </div>
    </div>
    <div v-if="expanded && files.length === 0 && phase.status === 'completed'" class="no-files">
      No output files
    </div>

    <!-- File content modal -->
    <n-modal v-model:show="showModal" preset="card" :title="selectedFile?.path" style="width: 800px; max-height: 80vh;">
      <n-spin v-if="loadingContent" />
      <n-scrollbar v-else style="max-height: 60vh;">
        <pre class="file-content">{{ fileContent }}</pre>
      </n-scrollbar>
    </n-modal>
  </n-card>
</template>

<style scoped>
.phase-card { transition: border-color 0.2s; }
.phase-header { display: flex; align-items: center; justify-content: space-between; }
.phase-title { font-weight: 600; font-size: 14px; }
.phase-meta { display: flex; gap: 16px; font-size: 12px; color: rgba(226, 232, 240, 0.6); margin-top: 4px; }
.meta-label { color: rgba(226, 232, 240, 0.35); margin-right: 4px; }
.file-list { margin-top: 10px; border-top: 1px solid rgba(226, 232, 240, 0.1); padding-top: 8px; }
.file-item { display: flex; align-items: center; gap: 8px; padding: 4px 8px; border-radius: 4px; font-size: 12px; color: rgba(226, 232, 240, 0.7); }
.file-item.clickable { cursor: pointer; }
.file-item.clickable:hover { background: rgba(0, 207, 200, 0.1); color: #00CFC8; }
.file-icon { font-size: 14px; }
.file-name { flex: 1; font-family: monospace; }
.file-size { color: rgba(226, 232, 240, 0.35); font-size: 11px; }
.phase-summary { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
.summary-row { display: flex; align-items: flex-start; gap: 8px; font-size: 12px; }
.summary-key { color: rgba(226, 232, 240, 0.5); min-width: 100px; text-transform: capitalize; flex-shrink: 0; padding-top: 2px; }
.summary-val { font-weight: 600; }
.summary-list { display: flex; flex-wrap: wrap; gap: 2px; }
.summary-link { color: #00CFC8; text-decoration: none; font-weight: 600; }
.summary-link:hover { text-decoration: underline; }
.summary-long { color: rgba(226, 232, 240, 0.8); word-break: break-word; line-height: 1.4; font-weight: normal; font-size: 12px; }
.accuracy-progress { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
.progress-item { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.progress-label { min-width: 70px; color: rgba(226, 232, 240, 0.5); flex-shrink: 0; }
.progress-text { font-size: 11px; color: rgba(226, 232, 240, 0.6); display: flex; gap: 4px; align-items: center; }
.no-files { font-size: 12px; color: rgba(226, 232, 240, 0.35); margin-top: 8px; }
.file-content { white-space: pre-wrap; word-break: break-all; font-size: 12px; font-family: monospace; line-height: 1.5; }
</style>
