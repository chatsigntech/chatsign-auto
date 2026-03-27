<script setup>
import { useI18n } from 'vue-i18n'
import StatusBadge from './StatusBadge.vue'
import { formatDate } from '../utils/format.js'

defineProps({ phase: Object })
const { t } = useI18n()
</script>

<template>
  <n-card size="small" class="phase-card">
    <div class="phase-header">
      <span class="phase-title">
        {{ t('task.phase') }} {{ phase.phase_num }} — {{ t(`phases.${phase.phase_num}`) }}
      </span>
      <StatusBadge :status="phase.status" />
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

    <n-alert v-if="phase.error_message" type="error" :title="t('task.errorMessage')" style="margin-top: 8px;">
      {{ phase.error_message }}
    </n-alert>
  </n-card>
</template>

<style scoped>
.phase-card {
  transition: border-color 0.2s;
}
.phase-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.phase-title {
  font-weight: 600;
  font-size: 14px;
}
.phase-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: rgba(226, 232, 240, 0.6);
  margin-top: 4px;
}
.meta-label {
  color: rgba(226, 232, 240, 0.35);
  margin-right: 4px;
}
</style>
