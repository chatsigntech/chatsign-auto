<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import AppHeader from '../components/AppHeader.vue'
import StatusBadge from '../components/StatusBadge.vue'
import {
  CheckmarkCircleOutline,
  CloseCircleOutline,
  TimeOutline,
  VideocamOutline
} from '@vicons/ionicons5'

const { t } = useI18n()
const { get } = useApi()

const status = ref(null)
const batches = ref([])
const selectedBatch = ref(null)
const loading = ref(false)

const videoCols = computed(() => [
  { title: 'Video ID', key: 'video_id', width: 160 },
  { title: t('accuracy.sentence'), key: 'sentence_text', ellipsis: { tooltip: true } },
  { title: t('accuracy.translator'), key: 'translator', width: 120 },
  { title: t('accuracy.filename'), key: 'filename', width: 240, ellipsis: { tooltip: true } },
])

async function fetchStatus(batch) {
  loading.value = true
  try {
    const url = batch ? `/api/accuracy/status?batch=${batch}` : '/api/accuracy/status'
    status.value = await get(url)
  } catch (e) {
    status.value = null
  } finally {
    loading.value = false
  }
}

async function fetchBatches() {
  try {
    const data = await get('/api/accuracy/batches')
    batches.value = data.batches
  } catch {}
}

function selectBatch(batch) {
  selectedBatch.value = batch
  fetchStatus(batch)
}

onMounted(() => {
  fetchStatus()
  fetchBatches()
})
</script>

<template>
  <div class="accuracy-page">
    <AppHeader />
    <div class="accuracy-content">
      <h2 class="page-title">{{ t('accuracy.title') }}</h2>

      <!-- Batch selector -->
      <div class="batch-bar">
        <n-button
          :type="!selectedBatch ? 'primary' : 'default'"
          :ghost="!!selectedBatch"
          size="small"
          @click="selectBatch(null)"
        >{{ t('accuracy.allBatches') }}</n-button>
        <n-button
          v-for="b in batches" :key="b.name"
          :type="selectedBatch === b.name ? 'primary' : 'default'"
          :ghost="selectedBatch !== b.name"
          size="small"
          @click="selectBatch(b.name)"
        >{{ b.name }} ({{ b.sentence_count }})</n-button>
      </div>

      <n-spin :show="loading">
        <template v-if="status">
          <!-- Summary cards -->
          <div class="summary-grid">
            <n-card class="summary-card">
              <div class="stat-icon"><n-icon :component="VideocamOutline" :size="28" color="#00CFC8" /></div>
              <div class="stat-value">{{ status.summary.total_submissions }}</div>
              <div class="stat-label">{{ t('accuracy.totalSubmissions') }}</div>
            </n-card>
            <n-card class="summary-card">
              <div class="stat-icon"><n-icon :component="CheckmarkCircleOutline" :size="28" color="#18A058" /></div>
              <div class="stat-value">{{ status.summary.approved }}</div>
              <div class="stat-label">{{ t('accuracy.approved') }}</div>
            </n-card>
            <n-card class="summary-card">
              <div class="stat-icon"><n-icon :component="CloseCircleOutline" :size="28" color="#D03050" /></div>
              <div class="stat-value">{{ status.summary.rejected }}</div>
              <div class="stat-label">{{ t('accuracy.rejected') }}</div>
            </n-card>
            <n-card class="summary-card">
              <div class="stat-icon"><n-icon :component="TimeOutline" :size="28" color="#F0A020" /></div>
              <div class="stat-value">{{ status.summary.pending_review }}</div>
              <div class="stat-label">{{ t('accuracy.pendingReview') }}</div>
            </n-card>
          </div>

          <!-- Ready indicator -->
          <n-alert
            v-if="status.summary.ready_for_pipeline"
            type="success"
            style="margin: 16px 0;"
          >{{ t('accuracy.readyMsg', { count: status.summary.approved }) }}</n-alert>
          <n-alert
            v-else
            type="warning"
            style="margin: 16px 0;"
          >{{ t('accuracy.notReadyMsg') }}</n-alert>

          <!-- Video lists -->
          <n-tabs type="line" style="margin-top: 16px;">
            <n-tab-pane :name="'approved'" :tab="`${t('accuracy.approved')} (${status.approved_videos.length})`">
              <n-data-table
                :columns="videoCols"
                :data="status.approved_videos"
                :max-height="400"
                size="small"
              />
            </n-tab-pane>
            <n-tab-pane :name="'pending'" :tab="`${t('accuracy.pendingReview')} (${status.pending_review.length})`">
              <n-data-table
                :columns="videoCols"
                :data="status.pending_review"
                :max-height="400"
                size="small"
              />
            </n-tab-pane>
            <n-tab-pane :name="'rejected'" :tab="`${t('accuracy.rejected')} (${status.rejected_videos.length})`">
              <n-data-table
                :columns="videoCols"
                :data="status.rejected_videos"
                :max-height="400"
                size="small"
              />
            </n-tab-pane>
          </n-tabs>
        </template>
        <n-empty v-else-if="!loading" :description="t('accuracy.noData')" style="margin-top: 80px;" />
      </n-spin>
    </div>
  </div>
</template>


<style scoped>
.accuracy-page { min-height: 100vh; }
.accuracy-content { max-width: 960px; margin: 0 auto; padding: 24px; }
.page-title { font-size: 22px; font-weight: 600; margin-bottom: 16px; }
.batch-bar { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}
.summary-card { text-align: center; }
.stat-icon { margin-bottom: 8px; }
.stat-value { font-size: 32px; font-weight: 700; color: #E2E8F0; }
.stat-label { font-size: 13px; color: rgba(226, 232, 240, 0.6); margin-top: 4px; }
</style>
