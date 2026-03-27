<script setup>
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'

const props = defineProps({ show: Boolean })
const emit = defineEmits(['update:show', 'created'])
const { t } = useI18n()
const { get, post } = useApi()

const name = ref('')
const preset = ref('medium')
const presets = ref([])
const loading = ref(false)
let presetsFetched = false

watch(() => props.show, async (visible) => {
  if (!visible || presetsFetched) return
  try {
    const data = await get('/api/config/presets')
    presets.value = data.presets.map(p => ({
      label: `${p.name} — ${p.description}`,
      value: p.name
    }))
    presetsFetched = true
  } catch {
    presets.value = [
      { label: 'light', value: 'light' },
      { label: 'medium', value: 'medium' },
      { label: 'heavy', value: 'heavy' }
    ]
  }
})

async function handleCreate() {
  if (!name.value.trim()) return
  loading.value = true
  try {
    const task = await post('/api/tasks/', { name: name.value.trim(), augmentation_preset: preset.value })
    name.value = ''
    preset.value = 'medium'
    emit('update:show', false)
    emit('created', task)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <n-modal
    :show="show"
    @update:show="emit('update:show', $event)"
    preset="card"
    :title="t('dashboard.create')"
    style="width: 480px;"
  >
    <n-space vertical :size="16">
      <n-input
        v-model:value="name"
        :placeholder="t('task.namePlaceholder')"
        @keyup.enter="handleCreate"
      />
      <n-select
        v-model:value="preset"
        :options="presets"
      />
      <n-space justify="end">
        <n-button @click="emit('update:show', false)">{{ t('task.cancel') }}</n-button>
        <n-button type="primary" :loading="loading" :disabled="!name.trim()" @click="handleCreate">
          {{ t('task.create') }}
        </n-button>
      </n-space>
    </n-space>
  </n-modal>
</template>
