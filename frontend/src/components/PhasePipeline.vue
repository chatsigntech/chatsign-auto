<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  CheckmarkCircle,
  EllipseOutline,
  PlayCircle,
  CloseCircle,
  PauseCircle
} from '@vicons/ionicons5'

const props = defineProps({
  phases: Array,
  currentPhase: Number
})
const { t } = useI18n()

const statusIcon = {
  pending: EllipseOutline,
  running: PlayCircle,
  completed: CheckmarkCircle,
  failed: CloseCircle,
  paused: PauseCircle
}

const statusColor = {
  pending: 'rgba(226, 232, 240, 0.3)',
  running: '#00CFC8',
  completed: '#18A058',
  failed: '#D03050',
  paused: '#F0A020'
}

const items = computed(() =>
  (props.phases || []).map(p => ({
    ...p,
    name: t(`phases.${p.phase_num}`),
    icon: statusIcon[p.status] || EllipseOutline,
    color: statusColor[p.status] || statusColor.pending,
    active: p.phase_num === props.currentPhase
  }))
)
</script>

<template>
  <div class="pipeline">
    <div v-for="(item, idx) in items" :key="item.phase_num" class="pipeline-step">
      <div class="step-node" :class="{ active: item.active }" :style="{ borderColor: item.color }">
        <n-icon :component="item.icon" :color="item.color" :size="18" />
        <span class="step-num">{{ item.phase_num }}</span>
      </div>
      <div class="step-label" :style="{ color: item.active ? '#00CFC8' : 'rgba(226,232,240,0.7)' }">
        {{ item.name }}
      </div>
      <div v-if="item.status === 'running'" class="step-progress">
        <n-progress
          type="line"
          :percentage="Math.round(item.progress)"
          :show-indicator="false"
          :height="4"
          color="#00CFC8"
        />
        <span class="progress-text">{{ Math.round(item.progress) }}%</span>
      </div>
      <div v-if="idx < items.length - 1" class="connector" :style="{ background: item.status === 'completed' ? '#18A058' : 'rgba(226,232,240,0.15)' }" />
    </div>
  </div>
</template>

<style scoped>
.pipeline {
  display: flex;
  align-items: flex-start;
  gap: 0;
  padding: 16px 0;
}
.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  flex: 1;
  min-width: 0;
}
.step-node {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 2px solid rgba(226, 232, 240, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  background: #1A1A2E;
  transition: all 0.3s;
  flex-shrink: 0;
}
.step-node.active {
  box-shadow: 0 0 12px rgba(0, 207, 200, 0.3);
}
.step-num {
  position: absolute;
  bottom: -2px;
  right: -4px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #252540;
  font-size: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(226, 232, 240, 0.6);
}
.step-label {
  margin-top: 6px;
  font-size: 10px;
  text-align: center;
  line-height: 1.2;
  max-width: 70px;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
}
.step-progress {
  margin-top: 4px;
  width: 60px;
  text-align: center;
}
.progress-text {
  font-size: 10px;
  color: #00CFC8;
}
.connector {
  position: absolute;
  top: 18px;
  left: calc(50% + 22px);
  width: calc(100% - 44px);
  height: 2px;
  z-index: 0;
}
</style>
