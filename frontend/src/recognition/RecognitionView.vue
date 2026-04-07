<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRecognition } from './useRecognition.js'
import { useTestVideo } from './useTestVideo.js'
import AppHeader from '../components/AppHeader.vue'
import {
  VideocamOutline,
  StopCircleOutline,
  RefreshOutline,
  PlayOutline,
  FlashOutline,
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
  startTestSession,
  stopSession,
  resetSession,
} = useRecognition()

const testVideo = useTestVideo()

const activeTab = ref('camera')
const cameraVideoRef = ref(null)
const testVideoRef = ref(null)

onMounted(() => {
  loadModels()
  testVideo.loadSteps()
})

onUnmounted(() => {
  stopSession()
  testVideo.reset()
})

// Camera tab handlers
function handleCameraStart() {
  if (cameraVideoRef.value) {
    startSession(cameraVideoRef.value)
  }
}

// Test video tab handlers
function handleGenerate() {
  if (selectedModel.value) {
    testVideo.generate(selectedModel.value)
  }
}

function handleTestStart() {
  if (testVideoRef.value && testVideo.videoUrl.value) {
    startTestSession(testVideoRef.value, testVideo.videoUrl.value)
  }
}

function handleTestStop() {
  stopSession()
  if (testVideoRef.value) {
    testVideoRef.value.pause()
    testVideoRef.value.currentTime = 0
  }
}

function handleTestTimeUpdate() {
  if (!testVideoRef.value) return
  const time = testVideoRef.value.currentTime
  testVideo.onTimeUpdate(time)
  // Auto-reset recognition session at sentence boundaries
  if (testVideo.checkBoundary()) {
    resetSession()
  }
}

// Stop session when switching tabs
watch(activeTab, () => {
  if (isStreaming.value || isModelLoading.value) {
    stopSession()
  }
})
</script>

<template>
  <AppHeader />
  <div class="recognition-page">
    <h2>{{ t('recognition.title') }}</h2>
    <p class="page-desc">{{ t('recognition.description') }}</p>

    <!-- Shared model selector -->
    <div class="controls">
      <n-select
        v-model:value="selectedModel"
        :options="modelOptions"
        :placeholder="t('recognition.selectModel')"
        :disabled="isStreaming || isModelLoading || testVideo.isGenerating.value"
        style="width: 360px"
        filterable
      />
    </div>

    <n-empty
      v-if="modelOptions.length === 0"
      :description="t('recognition.noModels')"
      style="margin-top: 60px"
    />

    <n-tabs v-if="modelOptions.length > 0" v-model:value="activeTab" type="line" animated>
      <!-- ==================== CAMERA TAB ==================== -->
      <n-tab-pane name="camera" :tab="t('recognition.liveCamera')">
        <div class="tab-controls">
          <n-button
            v-if="!isStreaming && !isModelLoading"
            type="primary"
            :disabled="!selectedModel"
            @click="handleCameraStart"
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

        <n-alert v-if="error && activeTab === 'camera'" type="error" closable style="margin-bottom: 16px" @close="error = null">
          {{ error }}
        </n-alert>

        <div v-if="selectedModel" class="main-content">
          <div class="camera-section">
            <n-card :title="t('recognition.camera')" size="small">
              <div class="video-container">
                <video ref="cameraVideoRef" autoplay playsinline muted class="video-feed" />
                <div v-if="isModelLoading" class="video-overlay">
                  <n-spin size="large" />
                  <span class="overlay-text">{{ t('recognition.loadingModel') }}</span>
                </div>
                <div v-if="!isStreaming && !isModelLoading" class="video-overlay">
                  <n-icon :component="VideocamOutline" :size="48" color="#555" />
                  <span class="overlay-text">{{ t('recognition.cameraOff') }}</span>
                </div>
              </div>
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

          <div class="results-section">
            <n-card :title="t('recognition.results')" size="small">
              <div class="token-list">
                <div v-if="results.length === 0" class="empty-results">
                  {{ t('recognition.waitingForSign') }}
                </div>
                <div v-for="(item, idx) in results" :key="idx" class="token-item">
                  <span class="token-text">{{ item.token }}</span>
                  <span v-if="item.score != null" class="token-score">{{ item.score.toFixed(2) }}</span>
                </div>
              </div>
              <div v-if="sentence" class="sentence-output">
                <div class="sentence-label">{{ t('recognition.sentence') }}</div>
                <div class="sentence-text">{{ sentence }}</div>
              </div>
            </n-card>
          </div>
        </div>
      </n-tab-pane>

      <!-- ==================== TEST VIDEO TAB ==================== -->
      <n-tab-pane name="test" :tab="t('recognition.testVideo')">
        <div class="tab-controls">
          <n-select
            v-model:value="testVideo.selectedStepKeys.value"
            :options="testVideo.stepOptions.value"
            :disabled="testVideo.isGenerating.value"
            multiple
            style="width: 400px"
            :placeholder="t('recognition.selectSteps')"
          />
          <n-button
            type="primary"
            :disabled="!selectedModel || testVideo.isGenerating.value"
            :loading="testVideo.isGenerating.value"
            @click="handleGenerate"
          >
            <template #icon><n-icon :component="FlashOutline" /></template>
            {{ testVideo.isGenerating.value ? t('recognition.generating') : t('recognition.generateTest') }}
          </n-button>
          <n-button
            v-if="testVideo.videoUrl.value && !isStreaming && !isModelLoading"
            type="info"
            @click="handleTestStart"
          >
            <template #icon><n-icon :component="PlayOutline" /></template>
            {{ t('recognition.startTest') }}
          </n-button>
          <n-button
            v-if="isStreaming || isModelLoading"
            type="error"
            @click="handleTestStop"
          >
            <template #icon><n-icon :component="StopCircleOutline" /></template>
            {{ t('recognition.stopTest') }}
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

        <!-- Progress bar during generation -->
        <div v-if="testVideo.isGenerating.value" style="margin-bottom: 16px">
          <n-progress
            type="line"
            :percentage="Math.round(testVideo.progress.value * 100)"
            :status="'active'"
          />
        </div>

        <n-alert v-if="testVideo.error.value" type="error" closable style="margin-bottom: 16px" @close="testVideo.error.value = null">
          {{ testVideo.error.value }}
        </n-alert>
        <n-alert v-if="error && activeTab === 'test'" type="error" closable style="margin-bottom: 16px" @close="error = null">
          {{ error }}
        </n-alert>

        <div v-if="testVideo.videoUrl.value" class="main-content">
          <!-- Video player -->
          <div class="camera-section">
            <n-card :title="t('recognition.videoPlayer')" size="small">
              <div class="video-container">
                <video
                  ref="testVideoRef"
                  class="video-feed"
                  muted
                  playsinline
                  @timeupdate="handleTestTimeUpdate"
                />
                <div v-if="isModelLoading" class="video-overlay">
                  <n-spin size="large" />
                  <span class="overlay-text">{{ t('recognition.loadingModel') }}</span>
                </div>
              </div>
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

          <!-- Per-sentence comparison -->
          <div class="results-section">
            <!-- Current sentence comparison -->
            <n-card :title="t('recognition.comparison')" size="small">
              <div v-if="testVideo.currentSentenceIndex.value < 0" class="empty-results">
                {{ t('recognition.waitingForSign') }}
              </div>
              <template v-else>
                <div class="comparison-row">
                  <span class="comparison-label gt-label">GT</span>
                  <span class="comparison-text gt-text">{{ testVideo.currentGtSentence.value }}</span>
                </div>
                <div class="comparison-row">
                  <span class="comparison-label pred-label">Pred</span>
                  <span class="comparison-text pred-text">{{ sentence || '...' }}</span>
                </div>
              </template>
            </n-card>

            <!-- Sentence timeline -->
            <n-card :title="t('recognition.timeline')" size="small" style="margin-top: 12px">
              <div class="sentence-list">
                <div
                  v-for="(s, idx) in testVideo.sentences.value"
                  :key="idx"
                  class="gt-sentence-item"
                  :class="{ active: idx === testVideo.currentSentenceIndex.value }"
                >
                  <div class="gt-sentence-header">
                    <n-tag
                      :type="idx === testVideo.currentSentenceIndex.value ? 'info' : 'default'"
                      size="small"
                    >
                      #{{ s.index }}
                    </n-tag>
                    <span class="gt-aug-name">{{ s.aug_desc }}</span>
                    <span class="gt-time">{{ s.start_time.toFixed(1) }}s - {{ s.end_time.toFixed(1) }}s</span>
                  </div>
                  <div class="gt-sentence-text">{{ s.sentence_text }}</div>
                </div>
              </div>
            </n-card>
          </div>
        </div>

        <!-- Empty state when no video generated -->
        <n-empty
          v-if="!testVideo.videoUrl.value && !testVideo.isGenerating.value"
          :description="t('recognition.noSentenceVideos')"
          style="margin-top: 60px"
        />
      </n-tab-pane>
    </n-tabs>
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
  margin-bottom: 16px;
}

.tab-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
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
  min-height: 160px;
  max-height: 240px;
  overflow-y: auto;
}

.empty-results {
  color: #555;
  text-align: center;
  padding: 40px 20px;
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

/* Per-sentence comparison */
.comparison-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 0;
}

.comparison-row + .comparison-row {
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.comparison-label {
  font-weight: 700;
  font-size: 13px;
  min-width: 40px;
  flex-shrink: 0;
  padding-top: 2px;
}

.gt-label {
  color: #18a058;
}

.pred-label {
  color: #00CFC8;
}

.comparison-text {
  font-size: 16px;
  line-height: 1.6;
}

.gt-text {
  color: #CBD5E1;
}

.pred-text {
  color: #E2E8F0;
  font-weight: 500;
}

/* Ground truth styles */
.sentence-list {
  max-height: 300px;
  overflow-y: auto;
}

.gt-sentence-item {
  padding: 8px 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  border-left: 3px solid transparent;
  transition: all 0.2s;
}

.gt-sentence-item.active {
  background: rgba(0, 207, 200, 0.08);
  border-left-color: #00CFC8;
}

.gt-sentence-item:last-child {
  border-bottom: none;
}

.gt-sentence-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.gt-aug-name {
  color: #888;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
}

.gt-time {
  color: #555;
  font-size: 12px;
  margin-left: auto;
}

.gt-sentence-text {
  color: #CBD5E1;
  font-size: 14px;
  line-height: 1.4;
}

.gt-sentence-item.active .gt-sentence-text {
  color: #E2E8F0;
  font-weight: 500;
}

@media (max-width: 768px) {
  .main-content {
    grid-template-columns: 1fr;
  }
}
</style>
