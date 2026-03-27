<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const props = defineProps({ status: String })
const { t } = useI18n()

const typeMap = {
  pending: 'default',
  running: 'info',
  completed: 'success',
  failed: 'error',
  paused: 'warning'
}

const tagType = computed(() => typeMap[props.status] || 'default')
const label = computed(() => t(`status.${props.status}`) || props.status)
</script>

<template>
  <n-tag :type="tagType" size="small" round>{{ label }}</n-tag>
</template>
