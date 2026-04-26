<template>
  <n-modal :show="true" preset="dialog" :title="modalTitle" :mask-closable="!busy">
    <!-- Form view -->
    <template v-if="!result">
      <n-alert v-if="pendingCount > 0" type="warning" style="margin-bottom: 12px;">
        Still {{ pendingCount }} videos pending review — only {{ approvedCount }} approved videos will be published.
      </n-alert>
      <n-alert v-else-if="approvedCount === 0" type="error" style="margin-bottom: 12px;">
        No approved videos to publish.
      </n-alert>
      <n-form size="small" label-placement="left" label-width="120">
        <n-form-item label="Host">
          <n-input v-model:value="form.host" placeholder="e.g. 1.2.3.4 or server.example.com" />
        </n-form-item>
        <n-form-item label="Port">
          <n-input-number v-model:value="form.port" :min="1" :max="65535" style="width: 100%;" />
        </n-form-item>
        <n-form-item label="Username">
          <n-input v-model:value="form.username" autocomplete="off" />
        </n-form-item>
        <n-form-item label="Password">
          <n-input v-model:value="form.password" type="password" show-password-on="click" autocomplete="new-password" />
        </n-form-item>
        <n-form-item label="Target dir">
          <n-input v-model:value="form.target_dir" placeholder="/data/foo (absolute path, must exist on remote)" />
        </n-form-item>
      </n-form>
    </template>

    <!-- Result view -->
    <template v-else>
      <p style="margin-bottom: 8px;">
        Uploaded <b>{{ result.success }}</b> / {{ result.total_videos }} videos.
        Gloss.csv: <b>{{ result.gloss_uploaded ? 'OK' : 'FAILED' }}</b>.
        Failed: <b>{{ result.failed }}</b>.
      </p>
      <p v-if="result.note" style="color: #888;">{{ result.note }}</p>
      <div v-if="result.errors && result.errors.length" style="max-height: 200px; overflow-y: auto;
           background: #fafafa; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 12px;">
        <div v-for="(e, i) in result.errors" :key="i" style="margin-bottom: 4px;">
          <b>{{ e.filename }}</b>: {{ e.msg }}
        </div>
      </div>
    </template>

    <template #action>
      <template v-if="!result">
        <n-button :disabled="busy" @click="onClose">Cancel</n-button>
        <n-button type="primary" :loading="busy" :disabled="!canSubmit" @click="submit">
          {{ busy ? 'Uploading…' : `Publish ${approvedCount} videos` }}
        </n-button>
      </template>
      <template v-else>
        <n-button type="primary" @click="onClose">OK</n-button>
      </template>
    </template>
  </n-modal>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useApi } from '../composables/useApi.js'

const props = defineProps({
  taskId: { type: String, required: true },
  pendingCount: { type: Number, default: 0 },
  approvedCount: { type: Number, default: 0 },
})
const emit = defineEmits(['close', 'done'])

const { post } = useApi()
const form = ref({ host: '', port: 22, username: '', password: '', target_dir: '' })
const busy = ref(false)
const result = ref(null)

const canSubmit = computed(() =>
  props.approvedCount > 0 &&
  form.value.host && form.value.username && form.value.password && form.value.target_dir
)

const modalTitle = computed(() => {
  if (!result.value) return 'Publish to remote'
  return result.value.failed === 0 ? '✓ Publish complete' : '⚠ Publish completed with errors'
})

async function submit() {
  busy.value = true
  try {
    const r = await post(`/api/tasks/${props.taskId}/phases/3/publish`, { ...form.value })
    result.value = r
    emit('done', r)
  } catch (e) {
    result.value = {
      success: 0, failed: -1, total_videos: props.approvedCount,
      gloss_uploaded: false,
      errors: [{ filename: '-', msg: (e?.message || String(e)).slice(0, 300) }],
    }
  } finally {
    // Always wipe password from in-memory state immediately
    form.value.password = ''
    busy.value = false
  }
}

function onClose() {
  // Wipe password just in case (e.g. user cancels mid-typing)
  form.value.password = ''
  emit('close')
}
</script>
