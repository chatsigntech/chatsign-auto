<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'

const props = defineProps({ show: Boolean })
const emit = defineEmits(['update:show', 'created'])
const { t } = useI18n()
const { post } = useApi()

const NAME_MAX = 50
const TEXT_MAX = 10000

const name = ref('')
const inputText = ref('')
const loading = ref(false)
const errorMsg = ref('')

async function handleCreate() {
  errorMsg.value = ''
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
  } catch (e) {
    errorMsg.value = e.message || t('task.createError')
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
      <n-alert v-if="errorMsg" type="error" closable @close="errorMsg = ''">
        {{ errorMsg }}
      </n-alert>
      <div>
        <label class="field-label">{{ t('task.name') }}</label>
        <n-input
          v-model:value="name"
          :placeholder="t('task.namePlaceholder')"
          :maxlength="NAME_MAX"
          show-count
        />
        <span class="field-hint">{{ t('task.nameHint', { max: NAME_MAX }) }}</span>
      </div>
      <div>
        <label class="field-label">{{ t('task.inputTextLabel') }}</label>
        <n-input
          v-model:value="inputText"
          type="textarea"
          :rows="4"
          :placeholder="t('task.inputTextPlaceholder')"
          :maxlength="TEXT_MAX"
          show-count
        />
        <span class="field-hint">{{ t('task.inputTextHint', { max: TEXT_MAX.toLocaleString() }) }}</span>
      </div>
      <n-space justify="end">
        <n-button @click="emit('update:show', false)">{{ t('task.cancel') }}</n-button>
        <n-button type="primary" :loading="loading" :disabled="!name.trim() || !inputText.trim()" @click="handleCreate">
          {{ t('task.create') }}
        </n-button>
      </n-space>
    </n-space>
  </n-modal>
</template>

<style scoped>
.field-label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 4px;
}
.field-hint {
  display: block;
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}
</style>
