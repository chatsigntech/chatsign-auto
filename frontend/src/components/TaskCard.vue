<script setup>
import { useI18n } from 'vue-i18n'
import StatusBadge from './StatusBadge.vue'
import { formatDate } from '../utils/format.js'

defineProps({ task: Object })
const { t } = useI18n()
</script>

<template>
  <n-card hoverable class="task-card" @click="$router.push(`/task/${task.task_id}`)">
    <div class="task-header">
      <span class="task-name">{{ task.name }}</span>
      <StatusBadge :status="task.status" />
    </div>
    <div class="task-meta">
      <span class="meta-item">
        <span class="meta-label">ID</span>
        <code>{{ task.task_id }}</code>
      </span>
      <span class="meta-item">
        <span class="meta-label">{{ t('task.currentPhase') }}</span>
        {{ t(`phases.${task.current_phase}`) }}
      </span>
      <span class="meta-item">
        <span class="meta-label">{{ t('task.createdAt') }}</span>
        {{ formatDate(task.created_at) }}
      </span>
    </div>
  </n-card>
</template>

<style scoped>
.task-card {
  cursor: pointer;
  transition: border-color 0.2s;
}
.task-card:hover {
  border-color: rgba(0, 207, 200, 0.4);
}
.task-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.task-name {
  font-size: 16px;
  font-weight: 600;
}
.task-meta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.meta-item {
  font-size: 13px;
  color: rgba(226, 232, 240, 0.7);
}
.meta-label {
  color: rgba(226, 232, 240, 0.45);
  margin-right: 6px;
}
code {
  font-size: 12px;
  color: #00CFC8;
}
</style>
