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

      <div class="picker-row">
        <span class="picker-label">Servers:</span>
        <n-select v-model:value="selectedNames" multiple
                  :options="serverOptions" :loading="loadingServers"
                  :placeholder="serverOptions.length ? 'Pick one or more' : 'No servers — click Manage to add'"
                  style="flex: 1;" />
        <router-link to="/publish-servers" target="_blank" class="manage-link"
                     @click="scheduleRefreshOnFocus">Manage</router-link>
      </div>
      <p v-if="loadServersError" class="hint">Failed to load servers: {{ loadServersError }}</p>
      <p v-else-if="!loadingServers && !serverOptions.length" class="hint">
        No servers configured. Click <b>Manage</b> to add one.
      </p>
    </template>

    <!-- Result view -->
    <template v-else>
      <p style="margin-bottom: 8px;">
        Published {{ result.total_videos }} approved videos to {{ result.per_server.length }} server(s).
        Overall: <b :style="{ color: result.overall_success ? '#18a058' : '#d03050' }">
          {{ result.overall_success ? 'success' : 'partial / failed' }}
        </b>
      </p>
      <div v-for="(s, i) in result.per_server" :key="i" class="server-result">
        <div class="server-result-head">
          <b>{{ s.name }}</b>
          <span :style="{ color: s.failed === 0 ? '#18a058' : '#d03050' }">
            {{ s.success }}/{{ s.total_videos }} ok
            <span v-if="s.failed > 0">· {{ s.failed }} failed</span>
            <span v-if="!s.gloss_uploaded">· gloss FAILED</span>
          </span>
        </div>
        <div v-if="s.note" class="server-note">{{ s.note }}</div>
        <details v-if="s.errors && s.errors.length" class="server-errors">
          <summary>{{ s.errors.length }} errors</summary>
          <div v-for="(e, j) in s.errors" :key="j" class="error-row">
            <b>{{ e.filename }}</b>: {{ e.msg }}
          </div>
        </details>
      </div>
    </template>

    <template #action>
      <template v-if="!result">
        <n-button :disabled="busy" @click="onClose">Cancel</n-button>
        <n-button type="primary" :loading="busy" :disabled="!canSubmit" @click="submit">
          {{ busy ? 'Uploading…' : `Publish to ${selectedNames.length || '?'} server(s)` }}
        </n-button>
      </template>
      <template v-else>
        <n-button type="primary" @click="onClose">OK</n-button>
      </template>
    </template>
  </n-modal>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useApi } from '../composables/useApi.js'

const props = defineProps({
  taskId: { type: String, required: true },
  pendingCount: { type: Number, default: 0 },
  approvedCount: { type: Number, default: 0 },
})
const emit = defineEmits(['close', 'published'])

const { get, post } = useApi()

const servers = ref([])
const selectedNames = ref([])
const loadingServers = ref(false)
const loadServersError = ref('')
const busy = ref(false)
const result = ref(null)

const serverOptions = computed(() =>
  servers.value.map(s => ({
    label: `${s.name}  —  ${s.username}@${s.host}:${s.port}${s.default_target_dir}`,
    value: s.name,
  }))
)

const canSubmit = computed(() =>
  props.approvedCount > 0 && selectedNames.value.length > 0
)

const modalTitle = computed(() => {
  if (!result.value) return 'Publish to remote'
  return result.value.overall_success ? '✓ Publish complete' : '⚠ Publish completed with errors'
})

async function loadServers() {
  loadingServers.value = true
  loadServersError.value = ''
  try {
    servers.value = await get('/api/publish-servers') || []
  } catch (e) {
    servers.value = []
    loadServersError.value = (e?.message || String(e)).slice(0, 120)
  } finally {
    loadingServers.value = false
  }
}

// When user clicks Manage and comes back, refresh ONCE then auto-unregister
// to avoid firing on unrelated focus events later.
let _focusHandler = null
function scheduleRefreshOnFocus() {
  if (_focusHandler) return
  _focusHandler = () => {
    loadServers()
    window.removeEventListener('focus', _focusHandler)
    _focusHandler = null
  }
  window.addEventListener('focus', _focusHandler)
}

async function submit() {
  busy.value = true
  try {
    const r = await post(`/api/tasks/${props.taskId}/phases/3/publish`, {
      server_names: selectedNames.value,
    })
    result.value = r
    emit('published')
  } catch (e) {
    result.value = {
      per_server: [{ name: '-', success: 0, failed: -1, total_videos: props.approvedCount,
                     gloss_uploaded: false,
                     errors: [{ filename: '-', msg: (e?.message || String(e)).slice(0, 300) }] }],
      overall_success: false,
      total_videos: props.approvedCount,
    }
  } finally {
    busy.value = false
  }
}

function onClose() {
  emit('close')
}

onMounted(loadServers)
onUnmounted(() => {
  if (_focusHandler) window.removeEventListener('focus', _focusHandler)
})
</script>

<style scoped>
.picker-row { display: flex; align-items: center; gap: 12px; }
.picker-label { font-size: 13px; color: #555; min-width: 60px; }
.manage-link { font-size: 13px; color: #00CFC8; text-decoration: none; white-space: nowrap; }
.manage-link:hover { text-decoration: underline; }
.hint { color: #888; font-size: 12px; margin-top: 6px; }
.server-result { background: #fafafa; padding: 8px 12px; border-radius: 4px; margin-bottom: 6px; }
.server-result-head { display: flex; justify-content: space-between; font-size: 13px; }
.server-note { color: #888; font-size: 12px; margin-top: 4px; }
.server-errors { margin-top: 6px; font-family: ui-monospace, monospace; font-size: 11px; }
.server-errors summary { cursor: pointer; color: #d03050; }
.error-row { padding: 2px 0; color: #555; }
</style>
