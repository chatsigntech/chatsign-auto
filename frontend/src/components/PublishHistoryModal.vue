<template>
  <n-modal :show="true" preset="card" title="Publish history"
           style="max-width: 720px;" @close="$emit('close')">
    <n-spin v-if="loading" />
    <p v-else-if="!history.length" class="empty">No publish attempts yet for this task.</p>
    <div v-else class="history-list">
      <div v-for="(r, i) in history" :key="i" class="history-row">
        <div class="row-head">
          <span class="ts">{{ formatTime(r.timestamp) }}</span>
          <span class="status" :class="{ ok: r.overall_success, fail: !r.overall_success }">
            {{ r.overall_success ? '✓ all ok' : '✗ partial / failed' }}
          </span>
        </div>
        <div class="row-body">
          <span class="meta">
            {{ r.total_videos }} videos →
            {{ (r.server_names || []).join(', ') }}
            <span v-if="r.user_id" class="user">by {{ r.user_id }}</span>
          </span>
          <details v-if="r.per_server && r.per_server.length" class="per-server">
            <summary>{{ r.per_server.length }} server result(s)</summary>
            <div v-for="(s, j) in r.per_server" :key="j" class="server-line">
              <b>{{ s.name }}</b>
              <span :class="{ ok: s.failed === 0, fail: s.failed !== 0 }">
                {{ s.success }}/{{ s.total_videos }} ok<span v-if="s.failed > 0">, {{ s.failed }} failed</span>
                <span v-if="!s.gloss_uploaded">, gloss FAILED</span>
              </span>
              <details v-if="s.errors && s.errors.length" class="errors-detail">
                <summary>{{ s.errors.length }} errors</summary>
                <div v-for="(e, k) in s.errors" :key="k" class="err">
                  <b>{{ e.filename }}</b>: {{ e.msg }}
                </div>
              </details>
            </div>
          </details>
        </div>
      </div>
    </div>
    <template #footer>
      <n-button @click="$emit('close')">Close</n-button>
    </template>
  </n-modal>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useApi } from '../composables/useApi.js'

const props = defineProps({ taskId: { type: String, required: true } })
defineEmits(['close'])

const { get } = useApi()
const history = ref([])
const loading = ref(true)

function formatTime(iso) {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    return d.toLocaleString()
  } catch { return iso }
}

onMounted(async () => {
  loading.value = true
  try {
    const r = await get(`/api/tasks/${props.taskId}/phases/3/publish-history`)
    history.value = Array.isArray(r) ? r : []
  } catch {
    history.value = []
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.empty { color: #888; padding: 16px; text-align: center; }
.history-list { max-height: 60vh; overflow-y: auto; }
.history-row { padding: 10px 12px; border-bottom: 1px solid #eee; }
.row-head { display: flex; justify-content: space-between; font-size: 12px; }
.ts { color: #555; font-family: ui-monospace, monospace; }
.status.ok { color: #18a058; }
.status.fail { color: #d03050; }
.row-body { margin-top: 4px; font-size: 13px; }
.meta { color: #555; }
.user { color: #888; margin-left: 6px; }
.per-server { margin-top: 6px; }
.per-server summary { cursor: pointer; color: #00CFC8; font-size: 12px; }
.server-line { padding: 4px 0 4px 16px; font-size: 12px; }
.server-line .ok { color: #18a058; margin-left: 8px; }
.server-line .fail { color: #d03050; margin-left: 8px; }
.errors-detail { margin-top: 4px; padding-left: 16px; }
.errors-detail summary { cursor: pointer; color: #d03050; font-size: 11px; }
.err { font-family: ui-monospace, monospace; font-size: 11px; padding: 2px 0; color: #555; }
</style>
