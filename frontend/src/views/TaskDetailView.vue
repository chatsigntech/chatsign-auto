<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useMessage, useDialog } from 'naive-ui'
import { useApi } from '../composables/useApi.js'
import { useTaskPolling } from '../composables/useTaskPolling.js'
import AppHeader from '../components/AppHeader.vue'
import StatusBadge from '../components/StatusBadge.vue'
import PhasePipeline from '../components/PhasePipeline.vue'
import PhaseCard from '../components/PhaseCard.vue'
import {
  PlayOutline,
  PauseOutline,
  PlaySkipForwardOutline,
  TrashOutline,
  ArrowBackOutline
} from '@vicons/ionicons5'
import { formatDate } from '../utils/format.js'

const route = useRoute()
const router = useRouter()
const { t } = useI18n()
const message = useMessage()
const dialog = useDialog()
const { get, post, del } = useApi()

const taskId = route.params.id
const { task, phases, loading, startPolling, fetchOnce } = useTaskPolling(taskId)
const actionLoading = ref(false)

const canRun = computed(() => ['pending', 'failed'].includes(task.value?.status))
const canPause = computed(() => task.value?.status === 'running')
const canResume = computed(() => task.value?.status === 'paused')
const canDelete = computed(() => ['pending', 'completed', 'failed', 'paused'].includes(task.value?.status))

async function runTask() {
  actionLoading.value = true
  try {
    await post(`/api/tasks/${taskId}/run`)
    message.success('Pipeline started')
    startPolling()
  } catch (e) {
    message.error(e.message)
  } finally {
    actionLoading.value = false
  }
}

async function pauseTask() {
  actionLoading.value = true
  try {
    await post(`/api/tasks/${taskId}/pause`)
    message.info('Pause signal sent')
    await fetchOnce()
  } catch (e) {
    message.error(e.message)
  } finally {
    actionLoading.value = false
  }
}

async function doResume() {
  actionLoading.value = true
  try {
    await post(`/api/tasks/${taskId}/resume`)
    message.success('Pipeline resumed')
    startPolling()
  } catch (e) {
    message.error(e.message)
  } finally {
    actionLoading.value = false
  }
}

async function resumeTask() {
  // If paused at Phase 2 (video collection), show confirmation with progress
  if (task.value?.current_phase === 2) {
    try {
      const progress = await get(`/api/tasks/${taskId}/phases/2/accuracy-progress`)
      const recorded = progress.recorded || 0
      const total = progress.total_glosses || 0
      const approved = progress.approved || 0
      const rejected = progress.rejected || 0
      const pending = progress.pending_review || 0

      dialog.warning({
        title: t('task.continuePhase'),
        content: `${t('accuracy.sentence')}: ${total}\n${t('accuracy.totalSubmissions')}: ${recorded}\n${t('accuracy.approved')}: ${approved}\n${t('accuracy.rejected')}: ${rejected}\n${t('accuracy.pendingReview')}: ${pending}\n\n${approved > 0 ? t('accuracy.readyMsg', { count: approved }) : t('accuracy.notReadyMsg')}`,
        positiveText: t('task.continuePhase'),
        negativeText: t('task.cancel'),
        onPositiveClick: doResume,
      })
    } catch {
      // If accuracy progress unavailable, resume directly
      doResume()
    }
  } else {
    doResume()
  }
}

function deleteTask() {
  dialog.warning({
    title: t('task.delete'),
    content: t('task.confirmDelete'),
    positiveText: t('task.delete'),
    negativeText: t('task.cancel'),
    onPositiveClick: async () => {
      try {
        await del(`/api/tasks/${taskId}`)
        message.success('Task deleted')
        router.push('/')
      } catch (e) {
        message.error(e.message)
      }
    }
  })
}

onMounted(startPolling)
</script>

<template>
  <div class="detail-page">
    <AppHeader />

    <div class="detail-content">
      <n-spin :show="loading && !task">
        <template v-if="task">
          <!-- Back + Actions -->
          <div class="detail-toolbar">
            <n-button quaternary @click="router.push('/')">
              <template #icon><n-icon :component="ArrowBackOutline" /></template>
              {{ t('nav.dashboard') }}
            </n-button>
            <n-space>
              <n-button v-if="canRun" type="primary" :loading="actionLoading" @click="runTask">
                <template #icon><n-icon :component="PlayOutline" /></template>
                {{ t('task.run') }}
              </n-button>
              <n-button v-if="canPause" type="warning" :loading="actionLoading" @click="pauseTask">
                <template #icon><n-icon :component="PauseOutline" /></template>
                {{ t('task.pause') }}
              </n-button>
              <!-- Resume button hidden: use per-phase "Continue" instead -->
              <n-button v-if="canDelete" type="error" ghost @click="deleteTask">
                <template #icon><n-icon :component="TrashOutline" /></template>
                {{ t('task.delete') }}
              </n-button>
            </n-space>
          </div>

          <!-- Task Info -->
          <n-card class="task-info-card">
            <div class="info-header">
              <h2 class="task-title">{{ task.name }}</h2>
              <StatusBadge :status="task.status" />
            </div>
            <div class="info-grid">
              <div class="info-item">
                <span class="info-label">ID</span>
                <code>{{ task.task_id }}</code>
              </div>
              <div class="info-item">
                <span class="info-label">{{ t('task.createdAt') }}</span>
                <span>{{ formatDate(task.created_at) }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">{{ t('task.updatedAt') }}</span>
                <span>{{ formatDate(task.updated_at) }}</span>
              </div>
            </div>
            <n-alert v-if="task.error_message" type="error" :title="t('task.errorMessage')" style="margin-top: 16px;">
              {{ task.error_message }}
            </n-alert>
          </n-card>

          <!-- Pipeline Visualization -->
          <n-card title="Pipeline" style="margin-top: 16px;">
            <PhasePipeline :phases="phases" :current-phase="task.current_phase" />
          </n-card>

          <!-- Phase Details -->
          <div class="phase-list">
            <PhaseCard v-for="p in phases" :key="p.phase_num" :phase="p" :task-id="task.task_id"
              :task-status="task.status" :current-phase="task.current_phase" @resume="resumeTask" />
          </div>
        </template>
      </n-spin>
    </div>
  </div>
</template>

<style scoped>
.detail-page {
  min-height: 100vh;
}
.detail-content {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px;
}
.detail-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}
.task-info-card {
  margin-bottom: 0;
}
.info-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
.task-title {
  font-size: 22px;
  font-weight: 600;
}
.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.info-item {
  font-size: 14px;
  color: rgba(226, 232, 240, 0.7);
}
.info-label {
  color: rgba(226, 232, 240, 0.4);
  margin-right: 8px;
}
code {
  color: #00CFC8;
  font-size: 13px;
}
.phase-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 16px;
}
</style>
