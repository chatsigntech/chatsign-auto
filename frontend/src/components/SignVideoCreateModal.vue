<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'

const props = defineProps({ show: Boolean })
const emit = defineEmits(['update:show', 'created'])

const { t } = useI18n()
const { post } = useApi()

const title = ref('')
const text = ref('')
const submitting = ref(false)

async function handleSubmit() {
  if (!title.value.trim() || !text.value.trim()) return
  submitting.value = true
  try {
    await post('/api/sign-video/generate', {
      title: title.value.trim(),
      text: text.value.trim(),
    })
    title.value = ''
    text.value = ''
    emit('update:show', false)
    emit('created')
  } catch {
    // handled by useApi
  } finally {
    submitting.value = false
  }
}

function handleClose() {
  emit('update:show', false)
}
</script>

<template>
  <n-modal :show="props.show" @update:show="handleClose" preset="card"
    :title="t('signVideo.create')" style="max-width: 560px;" :mask-closable="!submitting">
    <n-form @submit.prevent="handleSubmit">
      <n-form-item :label="t('signVideo.titleLabel')">
        <n-input v-model:value="title" :placeholder="t('signVideo.titlePlaceholder')"
          maxlength="100" show-count :disabled="submitting" />
      </n-form-item>
      <n-form-item :label="t('signVideo.textLabel')">
        <n-input v-model:value="text" type="textarea" :rows="5"
          :placeholder="t('signVideo.textPlaceholder')"
          maxlength="50000" show-count :disabled="submitting" />
      </n-form-item>
      <div style="display: flex; justify-content: flex-end; gap: 8px;">
        <n-button @click="handleClose" :disabled="submitting">{{ t('signVideo.cancel') }}</n-button>
        <n-button type="primary" @click="handleSubmit"
          :loading="submitting" :disabled="!title.value.trim() || !text.value.trim()">
          {{ t('signVideo.submit') }}
        </n-button>
      </div>
    </n-form>
  </n-modal>
</template>
