<script setup>
import { ref, reactive, computed, watch, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import StatusBadge from './StatusBadge.vue'
import { formatDate } from '../utils/format.js'

const props = defineProps({ phase: Object, taskId: String, taskStatus: String, currentPhase: Number })
const emit = defineEmits(['resume'])
const { t } = useI18n()
const { get } = useApi()

const isWaitingPhase = computed(() => props.taskStatus === 'paused' && props.phase.phase_num === props.currentPhase && props.phase.status !== 'running')
const isDatasetMode = computed(() => summary.value && summary.value.status === 'dataset')

const summary = ref(null)
const accuracyProgress = ref(null)

// Per-key expandable details
const expandedKey = ref(null)  // which summary key is currently expanded
const detailData = ref([])     // loaded detail items for the expanded key
const detailLoading = ref(false)

// Modals
const selectedFile = ref(null)
const fileContent = ref('')
const showModal = ref(false)
const loadingContent = ref(false)
const selectedVideo = ref(null)
const showVideoModal = ref(false)

function formatDuration(startStr, endStr) {
  if (!startStr) return ''
  const start = new Date(startStr)
  const end = endStr ? new Date(endStr) : new Date()
  const sec = Math.round((end - start) / 1000)
  if (sec < 60) return `${sec}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  return `${h}h ${m}m`
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

// Keys that can be expanded to show detail lists
const VIDEO_KEYS = new Set([
  'videos_collected', 'annotated_videos', 'preprocessed_videos',
  'output_videos', 'videos_generated', 'success',
  'transfer_success',
  '2d_cv', 'temporal', '3d_views', 'identity',
  'total_clips', 'output_clips',
])
const FILE_KEYS = new Set([
  'checkpoints', 'prototypes', 'poses_extracted', 'poses_filtered', 'poses_normalized', 'poses_corrupt',
  'segmented_videos', 'total_segments',
])
const TEXT_KEYS = new Set(['sentence_count', 'glosses_pushed', 'unique_sentences'])

function isExpandable(key, val) {
  if (typeof val === 'number' && val > 0) {
    return VIDEO_KEYS.has(key) || FILE_KEYS.has(key) || TEXT_KEYS.has(key)
  }
  return false
}

// Map Phase 6 aug keys to subdirectory names for video filtering (matched against rel_path)
const AUG_DIR_MAP = {
  '2d_cv': '/cv_aug/',
  'temporal': '/temporal_aug/',
  '3d_views': '/3d_views/',
  'identity': '/identity/',
}

async function toggleDetail(key) {
  if (expandedKey.value === key) {
    expandedKey.value = null
    detailData.value = []
    return
  }
  expandedKey.value = key
  detailData.value = []
  detailLoading.value = true

  try {
    if (TEXT_KEYS.has(key)) {
      // Load text content from phase output files
      let textFile = key === 'sentence_count' ? 'glosses.json'
        : key === 'glosses_pushed' ? 'glosses_upload.csv'
        : key === 'unique_sentences' ? 'sentences.txt' : null
      if (textFile) {
        const data = await get(`/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/files/${textFile}`)
        const content = data.content || ''
        if (textFile.endsWith('.json')) {
          try {
            const parsed = JSON.parse(content)
            // glosses.json: { sentence: [glosses] }
            detailData.value = Object.entries(parsed).map(([sent, glosses]) => ({
              _type: 'text', label: sent, detail: Array.isArray(glosses) ? glosses.join(', ') : glosses
            }))
          } catch { detailData.value = [{ _type: 'text', label: content.slice(0, 500) }] }
        } else {
          // CSV or plain text: one item per line
          detailData.value = content.split('\n').filter(l => l.trim()).map(l => ({ _type: 'text', label: l }))
        }
      }
    } else if (VIDEO_KEYS.has(key)) {
      const data = await get(`/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/videos`)
      let videos = data.videos || []
      // Filter by aug type subdirectory for Phase 8
      const dirPattern = AUG_DIR_MAP[key]
      if (dirPattern) {
        videos = videos.filter(v => (v.rel_path || '').includes(dirPattern))
      }
      detailData.value = videos.map(v => ({ ...v, _type: 'video' }))
    } else if (FILE_KEYS.has(key)) {
      const data = await get(`/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/files`)
      const filter = {
        checkpoints: f => f.path.includes('checkpoint') && f.path.endsWith('.pth'),
        prototypes: f => f.path.includes('prototype'),
        poses_extracted: f => f.path.includes('poses_raw') && f.path.endsWith('.pkl'),
        poses_filtered: f => f.path.includes('poses_filtered') && f.path.endsWith('.pkl'),
        poses_normalized: f => f.path.includes('poses_normed') && f.path.endsWith('.pkl'),
        poses_corrupt: f => f.path.includes('corrupt'),
      }
      const fn = filter[key] || (() => true)
      detailData.value = (data.files || []).filter(fn).map(f => ({ ...f, _type: 'file' }))
    }
  } catch { /* ignore */ }
  detailLoading.value = false
}

function playVideo(video) {
  const url = video.url || `/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/video/${video.filename}`
  selectedVideo.value = { ...video, streamUrl: url }
  showVideoModal.value = true
}

async function viewFile(file) {
  if (file.path.endsWith('.mp4') || file.path.endsWith('.webm')) {
    const videoName = file.path.split('/').pop()
    selectedVideo.value = {
      filename: videoName,
      sentence_text: videoName.replace(/\.\w+$/, ''),
      streamUrl: `/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/video/${videoName}`,
    }
    showVideoModal.value = true
    return
  }
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

// Summary + accuracy polling
async function loadSummary() {
  if (!props.taskId || summary.value) return
  try {
    const data = await get(`/api/tasks/${props.taskId}/phases/${props.phase.phase_num}/summary`)
    if (data && Object.keys(data).length > 0) summary.value = data
  } catch { /* ignore */ }
}

let accuracyPollTimer = null
async function loadAccuracyProgress() {
  if (!props.taskId || props.phase.phase_num !== 2) return
  try {
    const data = await get(`/api/tasks/${props.taskId}/phases/2/accuracy-progress`)
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

let summaryPollTimer = null
function startSummaryPolling() {
  if (summaryPollTimer) return
  loadSummary()
  summaryPollTimer = setInterval(() => { summary.value = null; loadSummary() }, 15000)
}
function stopSummaryPolling() {
  if (summaryPollTimer) { clearInterval(summaryPollTimer); summaryPollTimer = null }
}

watch(() => props.phase?.status, (s) => {
  if (s === 'completed') { summary.value = null; loadSummary(); stopSummaryPolling(); stopAccuracyPolling() }
  if (s === 'running') { startSummaryPolling() }
  if (props.phase.phase_num === 2 && (s === 'pending' || s === 'running') && !isDatasetMode.value) { startAccuracyPolling() }
}, { immediate: true })

// Stop accuracy polling once we know this is dataset mode
watch(isDatasetMode, (v) => { if (v) stopAccuracyPolling() })

onUnmounted(() => { stopAccuracyPolling(); stopSummaryPolling() })
</script>

<template>
  <n-card size="small" class="phase-card">
    <!-- Header -->
    <div class="phase-header">
      <span class="phase-title">
        {{ t('task.phase') }} {{ phase.phase_num }} — {{ t(`phases.${phase.phase_num}`) }}
      </span>
      <StatusBadge :status="phase.status" />
    </div>

    <n-progress
      v-if="phase.status === 'running' || phase.progress > 0"
      type="line" :percentage="Math.round(phase.progress)" :height="6"
      color="#00CFC8" style="margin: 8px 0;"
    />

    <div class="phase-meta">
      <span v-if="phase.started_at"><span class="meta-label">Start</span> {{ formatDate(phase.started_at) }}</span>
      <span v-if="phase.completed_at"><span class="meta-label">End</span> {{ formatDate(phase.completed_at) }}</span>
      <span v-if="phase.started_at && (phase.completed_at || phase.status === 'running')">
        <span class="meta-label">Duration</span>
        <span :style="{ color: phase.status === 'running' ? '#00CFC8' : '' }">{{ formatDuration(phase.started_at, phase.completed_at) }}</span>
      </span>
      <span v-if="phase.gpu_id != null"><span class="meta-label">{{ t('task.gpuId') }}</span> {{ phase.gpu_id }}</span>
    </div>

    <!-- Phase 2: accuracy link + live progress (hidden in dataset mode) -->
    <div v-if="phase.phase_num === 2 && !isDatasetMode" class="phase-summary" style="margin-top: 8px;">
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

    <!-- Summary: each stat is independently expandable -->
    <div v-if="summary" class="phase-summary">
      <template v-for="(val, key) in summary" :key="key">
        <div class="summary-row">
          <span class="summary-key">{{ key.replace(/_/g, ' ') }}</span>
          <!-- Array: file names with download links -->
          <span v-if="Array.isArray(val) && key === 'dataset_files'" class="summary-val summary-list">
            <a v-for="item in val" :key="item" class="dl-tag"
              :href="`/api/tasks/${taskId}/phases/${phase.phase_num}/download/${item}`" :download="item">
              <n-tag size="tiny" :bordered="false" type="info" style="margin: 1px; cursor: pointer;">{{ item }} ↓</n-tag>
            </a>
          </span>
          <!-- Array (glosses list) -->
          <span v-else-if="Array.isArray(val)" class="summary-val summary-list">
            <n-tag v-for="item in val" :key="item" size="tiny" :bordered="false" type="info" style="margin: 1px;">{{ item }}</n-tag>
            <a v-if="key === 'glosses'" class="dl-link"
              :href="`/api/tasks/${taskId}/phases/${phase.phase_num}/download/glosses.json`" download>↓</a>
          </span>
          <!-- Link -->
          <a v-else-if="typeof val === 'string' && val.startsWith('http')" :href="val" target="_blank" class="summary-val summary-link">{{ val }}</a>
          <!-- Long text -->
          <span v-else-if="typeof val === 'string' && val.length > 60" class="summary-val summary-long">{{ val }}</span>
          <!-- Expandable count (videos / files) -->
          <n-tag v-else-if="isExpandable(key, val)"
            size="small" :bordered="false" type="success" style="cursor: pointer;" @click.stop="toggleDetail(key)">
            <span class="summary-val">{{ val }} {{ expandedKey === key ? '▲' : '▼' }}</span>
          </n-tag>
          <!-- Plain value -->
          <n-tag v-else size="small" :bordered="false" :type="typeof val === 'number' && val > 0 ? 'success' : 'default'">
            <span class="summary-val">{{ val }}</span>
          </n-tag>
        </div>

        <!-- Detail list for this key (inline, right below the row) -->
        <div v-if="expandedKey === key" class="detail-list">
          <n-spin v-if="detailLoading" size="small" style="margin: 8px 0;" />
          <template v-else-if="detailData.length > 0">
            <!-- Text items (sentences, glosses) -->
            <template v-if="detailData[0]._type === 'text'">
              <div v-for="(item, i) in detailData" :key="i" class="text-item">
                <span class="text-label">{{ item.label }}</span>
                <span v-if="item.detail" class="text-detail">{{ item.detail }}</span>
              </div>
            </template>
            <!-- Video items -->
            <template v-else-if="detailData[0]._type === 'video'">
              <div v-for="item in detailData" :key="item.filename" class="video-item">
                <span class="video-icon" @click="playVideo(item)" style="cursor:pointer;">🎬</span>
                <span class="video-gloss" @click="playVideo(item)" style="cursor:pointer;">{{ item.sentence_text || item.filename }}</span>
                <n-tag v-for="g in (item.glosses || [])" :key="g" size="tiny" :bordered="false" type="info" style="margin: 0 2px;">{{ g }}</n-tag>
                <n-tag v-if="item.preprocessed" size="tiny" :bordered="false" type="success" style="margin: 0 2px;">576p</n-tag>
                <span class="video-filename" @click="playVideo(item)" style="cursor:pointer;">{{ item.filename }}</span>
                <span class="video-size">{{ formatSize(item.size) }}</span>
                <a class="dl-link" :href="item.url || `/api/tasks/${taskId}/phases/${phase.phase_num}/video/${item.filename}`" :download="item.filename" @click.stop>↓</a>
              </div>
            </template>
            <!-- File items -->
            <template v-else>
              <div v-for="item in detailData" :key="item.path" class="file-item"
                :class="{ clickable: item.is_text }" @click="viewFile(item)">
                <span class="file-icon">{{ item.is_text ? '📄' : '📦' }}</span>
                <span class="file-name">{{ item.path.split('/').pop() }}</span>
                <span class="file-size">{{ formatSize(item.size) }}</span>
                <a class="dl-link" :href="`/api/tasks/${taskId}/phases/${phase.phase_num}/download/${item.path}`" :download="item.path.split('/').pop()" @click.stop>↓</a>
              </div>
            </template>
          </template>
          <div v-else class="no-files">No items</div>
        </div>
      </template>
    </div>

    <!-- Continue button -->
    <div v-if="isWaitingPhase" style="margin-top: 10px;">
      <n-button type="primary" size="small" @click="emit('resume')">
        {{ t('task.continuePhase') || 'Complete & Continue' }}
      </n-button>
    </div>

    <n-alert v-if="phase.error_message" type="error" :title="t('task.errorMessage')" style="margin-top: 8px;">
      {{ phase.error_message }}
    </n-alert>

    <!-- File content modal -->
    <n-modal v-model:show="showModal" preset="card" :title="selectedFile?.path" style="width: 800px; max-height: 80vh;">
      <n-spin v-if="loadingContent" />
      <n-scrollbar v-else style="max-height: 60vh;">
        <pre class="file-content">{{ fileContent }}</pre>
      </n-scrollbar>
    </n-modal>

    <!-- Video player modal -->
    <n-modal v-model:show="showVideoModal" preset="card"
      :title="selectedVideo ? (selectedVideo.sentence_text || selectedVideo.filename) : ''"
      style="width: 480px;">
      <template v-if="selectedVideo">
        <video :src="selectedVideo.streamUrl" controls autoplay
          style="width: 100%; border-radius: 4px; background: #000;" />
        <div style="margin-top: 8px; text-align: right;">
          <a class="dl-btn" :href="selectedVideo.streamUrl" :download="selectedVideo.filename">Download</a>
        </div>
      </template>
    </n-modal>
  </n-card>
</template>

<style scoped>
.phase-card { transition: border-color 0.2s; }
.phase-header { display: flex; align-items: center; justify-content: space-between; }
.phase-title { font-weight: 600; font-size: 14px; }
.phase-meta { display: flex; gap: 16px; font-size: 12px; color: rgba(226, 232, 240, 0.6); margin-top: 4px; }
.meta-label { color: rgba(226, 232, 240, 0.35); margin-right: 4px; }
.phase-summary { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
.summary-row { display: flex; align-items: flex-start; gap: 8px; font-size: 12px; }
.summary-key { color: rgba(226, 232, 240, 0.5); min-width: 120px; text-transform: capitalize; flex-shrink: 0; padding-top: 2px; }
.summary-val { font-weight: 600; }
.summary-list { display: flex; flex-wrap: wrap; gap: 2px; }
.summary-link { color: #00CFC8; text-decoration: none; font-weight: 600; }
.summary-link:hover { text-decoration: underline; }
.summary-long { color: rgba(226, 232, 240, 0.8); word-break: break-word; line-height: 1.4; font-weight: normal; font-size: 12px; }
.accuracy-progress { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
.progress-item { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.progress-label { min-width: 70px; color: rgba(226, 232, 240, 0.5); flex-shrink: 0; }
.progress-text { font-size: 11px; color: rgba(226, 232, 240, 0.6); display: flex; gap: 4px; align-items: center; }
.no-files { font-size: 12px; color: rgba(226, 232, 240, 0.35); margin: 4px 0 4px 128px; }

/* Detail list (inline under summary row) */
.detail-list { margin: 2px 0 6px 128px; border-left: 2px solid rgba(0, 207, 200, 0.2); padding-left: 8px; }

/* Video items */
.video-item { display: flex; align-items: center; gap: 8px; padding: 4px 8px; border-radius: 4px; font-size: 12px; color: rgba(226, 232, 240, 0.7); cursor: pointer; }
.video-item:hover { background: rgba(0, 207, 200, 0.1); color: #00CFC8; }
.video-icon { font-size: 14px; flex-shrink: 0; }
.video-gloss { font-weight: 600; min-width: 80px; }
.video-filename { flex: 1; font-family: monospace; color: rgba(226, 232, 240, 0.4); font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.video-size { color: rgba(226, 232, 240, 0.35); font-size: 11px; flex-shrink: 0; }

/* File items */
.file-item { display: flex; align-items: center; gap: 8px; padding: 3px 8px; border-radius: 4px; font-size: 12px; color: rgba(226, 232, 240, 0.7); }
.file-item.clickable { cursor: pointer; }
.file-item.clickable:hover { background: rgba(0, 207, 200, 0.1); color: #00CFC8; }
.file-icon { font-size: 14px; }
.file-name { flex: 1; font-family: monospace; }
.file-size { color: rgba(226, 232, 240, 0.35); font-size: 11px; }
.file-content { white-space: pre-wrap; word-break: break-all; font-size: 12px; font-family: monospace; line-height: 1.5; }
/* Text items (sentences, glosses) */
.text-item { display: flex; align-items: baseline; gap: 8px; padding: 3px 8px; font-size: 12px; color: rgba(226, 232, 240, 0.7); }
.text-label { flex: 1; }
.text-detail { color: #00CFC8; font-weight: 600; font-size: 11px; }
.dl-tag { text-decoration: none; }
.dl-link { color: rgba(0, 207, 200, 0.6); text-decoration: none; font-size: 12px; flex-shrink: 0; padding: 0 4px; }
.dl-link:hover { color: #00CFC8; }
.dl-btn { color: #00CFC8; text-decoration: none; font-size: 13px; font-weight: 600; }
.dl-btn:hover { text-decoration: underline; }
</style>
