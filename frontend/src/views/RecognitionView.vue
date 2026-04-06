<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRecognition } from '../composables/useRecognition.js'
import AppHeader from '../components/AppHeader.vue'
import {
  VideocamOutline,
  StopCircleOutline,
  RefreshOutline
} from '@vicons/ionicons5'

const { t } = useI18n()

const {
  isConnected,
  isStreaming,
  isModelLoading,
  selectedModel,
  modelOptions,
  results,
  sentence,
  stats,
  error,
  loadModels,
  startSession,
  stopSession,
  resetSession,
} = useRecognition()

const videoRef = ref(null)

onMounted(() => { loadModels() })

onUnmounted(() => { stopSession() })

function handleStart() {
  if (videoRef.value) {
    startSession(videoRef.value)
  }
}
</script>

<template>
  <AppHeader />
  <div class="recognition-page">
    <h2>{{ t('recognition.title') }}</h2>
    <p class="page-desc">{{ t('recognition.description') }}</p>

    <!-- Controls -->
    <div class="controls">
      <n-select
        v-model:value="selectedModel"
        :options="modelOptions"
        :placeholder="t('recognition.selectModel')"
        :disabled="isStreaming || isModelLoading"
        style="width: 360px"
        filterable
      />
      <n-button
        v-if="!isStreaming && !isModelLoading"
        type="primary"
        :disabled="!selectedModel"
        @click="handleStart"
      >
        <template #icon><n-icon :component="VideocamOutline" /></template>
        {{ t('recognition.start') }}
      </n-button>
      <n-button
        v-if="isStreaming || isModelLoading"
        type="error"
        @click="stopSession"
      >
        <template #icon><n-icon :component="StopCircleOutline" /></template>
        {{ t('recognition.stop') }}
      </n-button>
      <n-button
        v-if="isStreaming"
        tertiary
        @click="resetSession"
      >
        <template #icon><n-icon :component="RefreshOutline" /></template>
        {{ t('recognition.reset') }}
      </n-button>
    </div>

    <!-- Error -->
    <n-alert
      v-if="error"
      type="error"
      closable
      style="margin-bottom: 16px"
      @close="error = null"
    >
      {{ error }}
    </n-alert>

    <!-- Empty state -->
    <n-empty
      v-if="modelOptions.length === 0 && !isStreaming"
      :description="t('recognition.noModels')"
      style="margin-top: 60px"
    />

    <!-- Main content -->
    <div v-if="selectedModel" class="main-content">
      <!-- Camera feed -->
      <div class="camera-section">
        <n-card :title="t('recognition.camera')" size="small">
          <div class="video-container">
            <video
              ref="videoRef"
              autoplay
              playsinline
              muted
              class="video-feed"
            />
            <div v-if="isModelLoading" class="video-overlay">
              <n-spin size="large" />
              <span class="overlay-text">{{ t('recognition.loadingModel') }}</span>
            </div>
            <div v-if="!isStreaming && !isModelLoading" class="video-overlay">
              <n-icon :component="VideocamOutline" :size="48" color="#555" />
              <span class="overlay-text">{{ t('recognition.cameraOff') }}</span>
            </div>
          </div>

          <!-- Status bar -->
          <div class="status-bar">
            <n-tag :type="isConnected ? 'success' : 'default'" size="small">
              {{ isConnected ? t('recognition.connected') : t('recognition.disconnected') }}
            </n-tag>
            <span v-if="isStreaming" class="stat-text">
              {{ t('recognition.poseFrames') }}: {{ stats.pose_frames }}
              &nbsp;|&nbsp;
              {{ t('recognition.windows') }}: {{ stats.windows }}
            </span>
          </div>
        </n-card>
      </div>

      <!-- Results -->
      <div class="results-section">
        <n-card :title="t('recognition.results')" size="small">
          <div class="token-list">
            <div v-if="results.length === 0" class="empty-results">
              {{ t('recognition.waitingForSign') }}
            </div>
            <div
              v-for="(item, idx) in results"
              :key="idx"
              class="token-item"
            >
              <span class="token-text">{{ item.token }}</span>
              <span v-if="item.score != null" class="token-score">
                {{ item.score.toFixed(2) }}
              </span>
            </div>
          </div>

          <!-- Sentence output -->
          <div v-if="sentence" class="sentence-output">
            <div class="sentence-label">{{ t('recognition.sentence') }}</div>
            <div class="sentence-text">{{ sentence }}</div>
          </div>
        </n-card>
      </div>
    </div>
  </div>
</template>

<style scoped>
.recognition-page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 32px;
}

h2 {
  color: #E2E8F0;
  margin: 0 0 4px;
}

.page-desc {
  color: #888;
  font-size: 14px;
  margin: 0 0 20px;
}

.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.main-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.video-container {
  position: relative;
  width: 100%;
  aspect-ratio: 4 / 3;
  background: #111;
  border-radius: 6px;
  overflow: hidden;
}

.video-feed {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.video-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: rgba(0, 0, 0, 0.7);
}

.overlay-text {
  color: #888;
  font-size: 14px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
}

.stat-text {
  color: #888;
  font-size: 13px;
}

.token-list {
  min-height: 240px;
  max-height: 340px;
  overflow-y: auto;
}

.empty-results {
  color: #555;
  text-align: center;
  padding: 60px 20px;
  font-size: 14px;
}

.token-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.token-item:last-child {
  border-bottom: none;
}

.token-text {
  font-weight: 600;
  color: #00CFC8;
  font-size: 15px;
  font-family: 'JetBrains Mono', monospace;
}

.token-score {
  color: #888;
  font-size: 13px;
  font-family: 'JetBrains Mono', monospace;
}

.sentence-output {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.sentence-label {
  color: #888;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}

.sentence-text {
  color: #E2E8F0;
  font-size: 16px;
  line-height: 1.5;
}

@media (max-width: 768px) {
  .main-content {
    grid-template-columns: 1fr;
  }
}
</style>
