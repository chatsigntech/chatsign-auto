<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import { SearchOutline } from '@vicons/ionicons5'

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

// Suggest sentences
const topic = ref('')
const sentenceCount = ref(50)
const suggesting = ref(false)
const datasetVideos = ref(null) // null = user-typed, array = from dataset

async function handleSuggest() {
  errorMsg.value = ''
  if (!topic.value.trim()) return
  suggesting.value = true
  try {
    const data = await post('/api/tasks/suggest-sentences', {
      topic: topic.value.trim(),
      count: sentenceCount.value,
    })
    if (data.details && data.details.length) {
      inputText.value = data.details.map(d => d.text).join('. ') + '.'
      datasetVideos.value = data.details.map(d => ({
        text: d.text,
        vid: d.vid,
        source: d.source,
      }))
    } else {
      errorMsg.value = t('task.suggestEmpty')
      datasetVideos.value = null
    }
  } catch (e) {
    errorMsg.value = e.message || t('task.suggestError')
    datasetVideos.value = null
  } finally {
    suggesting.value = false
  }
}

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
    if (datasetVideos.value) {
      body.source = 'dataset'
      body.dataset_videos = datasetVideos.value
    }
    const task = await post('/api/tasks/', body)
    name.value = ''
    inputText.value = ''
    topic.value = ''
    datasetVideos.value = null
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
    style="width: 560px;"
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

      <!-- Suggest sentences -->
      <div class="suggest-section">
        <label class="field-label">{{ t('task.suggestLabel') }}</label>
        <div class="suggest-row">
          <n-input
            v-model:value="topic"
            :placeholder="t('task.suggestPlaceholder')"
            class="suggest-topic"
            :disabled="suggesting"
          />
          <n-input-number
            v-model:value="sentenceCount"
            :min="10"
            :max="200"
            :step="10"
            style="width: 100px;"
            :disabled="suggesting"
          />
          <n-button
            :loading="suggesting"
            :disabled="!topic.trim()"
            @click="handleSuggest"
          >
            <template #icon><n-icon :component="SearchOutline" /></template>
            {{ t('task.suggestBtn') }}
          </n-button>
        </div>
        <span class="field-hint">{{ t('task.suggestHint') }}</span>
      </div>

      <div>
        <label class="field-label">{{ t('task.inputTextLabel') }}</label>
        <n-input
          v-model:value="inputText"
          type="textarea"
          :rows="6"
          :placeholder="t('task.inputTextPlaceholder')"
          :maxlength="TEXT_MAX"
          show-count
          @input="datasetVideos = null"
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
.suggest-section {
  background: #f9f9fb;
  border-radius: 8px;
  padding: 12px;
}
.suggest-row {
  display: flex;
  gap: 8px;
  align-items: center;
}
.suggest-topic {
  flex: 1;
}
</style>
