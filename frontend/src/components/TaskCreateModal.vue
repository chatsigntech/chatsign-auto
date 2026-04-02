<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'

const props = defineProps({ show: Boolean })
const emit = defineEmits(['update:show', 'created'])
const { t } = useI18n()
const { post } = useApi()

const name = ref('')
const inputText = ref('')
const loading = ref(false)

async function handleCreate() {
  if (!name.value.trim() || !inputText.value.trim()) return
  loading.value = true
  try {
    const body = {
      name: name.value.trim(),
      input_text: inputText.value.trim(),
      batch_name: name.value.trim(),
    }
    const task = await post('/api/tasks/', body)
    name.value = ''
    inputText.value = ''
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
      />
      <n-input
        v-model:value="inputText"
        type="textarea"
        :rows="3"
        :placeholder="t('task.inputTextPlaceholder')"
      />
      <n-space justify="end">
        <n-button @click="emit('update:show', false)">{{ t('task.cancel') }}</n-button>
        <n-button type="primary" :loading="loading" :disabled="!name.trim() || !inputText.trim()" @click="handleCreate">
          {{ t('task.create') }}
        </n-button>
      </n-space>
    </n-space>
  </n-modal>
</template>
