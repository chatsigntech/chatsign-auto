<template>
  <div class="page">
    <div class="header">
      <h2>Publish Servers</h2>
      <n-button type="primary" size="small" @click="openAdd">+ Add server</n-button>
    </div>

    <n-spin v-if="loading" />
    <p v-else-if="!servers.length" class="empty">No servers yet. Click "+ Add server" to create one.</p>

    <table v-else class="server-table">
      <thead>
        <tr><th>Name</th><th>Host</th><th>Port</th><th>User</th><th>Target dir</th><th></th></tr>
      </thead>
      <tbody>
        <tr v-for="s in servers" :key="s.name">
          <td>{{ s.name }}</td>
          <td>{{ s.host }}</td>
          <td>{{ s.port }}</td>
          <td>{{ s.username }}</td>
          <td><code>{{ s.default_target_dir }}</code></td>
          <td class="actions">
            <n-button size="tiny" @click="openEdit(s)">Edit</n-button>
            <n-button size="tiny" type="error" @click="confirmDelete(s)">Delete</n-button>
          </td>
        </tr>
      </tbody>
    </table>

    <!-- Add/Edit modal -->
    <n-modal :show="showForm" preset="dialog" :title="editingName ? `Edit: ${editingName}` : 'Add server'"
             :mask-closable="false">
      <n-form size="small" label-placement="left" label-width="120">
        <n-form-item label="Name" v-if="!editingName">
          <n-input v-model:value="form.name" placeholder="unique label, e.g. prod-storage" />
        </n-form-item>
        <n-form-item label="Host">
          <n-input v-model:value="form.host" placeholder="1.2.3.4 or server.example.com" />
        </n-form-item>
        <n-form-item label="Port">
          <n-input-number v-model:value="form.port" :min="1" :max="65535" style="width: 100%;" />
        </n-form-item>
        <n-form-item label="Username">
          <n-input v-model:value="form.username" autocomplete="off" />
        </n-form-item>
        <n-form-item :label="editingName ? 'Password (leave blank to keep)' : 'Password'">
          <n-input v-model:value="form.password" type="password" show-password-on="click" autocomplete="new-password" />
        </n-form-item>
        <n-form-item label="Target dir">
          <n-input v-model:value="form.default_target_dir" placeholder="/data/foo (absolute path)" />
        </n-form-item>
      </n-form>
      <p v-if="formError" class="form-error">{{ formError }}</p>
      <template #action>
        <n-button :disabled="busy" @click="closeForm">Cancel</n-button>
        <n-button type="primary" :loading="busy" :disabled="!canSubmit" @click="submitForm">
          {{ editingName ? 'Save' : 'Add' }}
        </n-button>
      </template>
    </n-modal>

    <!-- Delete confirmation -->
    <n-modal :show="!!pendingDelete" preset="dialog" type="warning"
             title="Delete server" :positive-text="`Delete ${pendingDelete?.name || ''}`"
             negative-text="Cancel"
             @positive-click="doDelete" @negative-click="pendingDelete = null"
             @mask-click="pendingDelete = null">
      Are you sure you want to delete <b>{{ pendingDelete?.name }}</b>? This cannot be undone.
    </n-modal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useApi } from '../composables/useApi.js'

const { get, post, put, del } = useApi()

const servers = ref([])
const loading = ref(true)

const showForm = ref(false)
const editingName = ref(null)   // null = adding; otherwise the name being edited
const form = ref(_emptyForm())
const busy = ref(false)
const formError = ref('')
const pendingDelete = ref(null)

function _emptyForm() {
  return { name: '', host: '', port: 22, username: '', password: '', default_target_dir: '' }
}

const canSubmit = computed(() => {
  const f = form.value
  if (!f.host || !f.username || !f.default_target_dir) return false
  if (editingName.value) return true   // editing — password optional
  return !!f.name && !!f.password      // adding — name + password required
})

async function load() {
  loading.value = true
  try {
    servers.value = await get('/api/publish-servers') || []
  } catch (e) {
    servers.value = []
  } finally {
    loading.value = false
  }
}

function openAdd() {
  editingName.value = null
  form.value = _emptyForm()
  formError.value = ''
  showForm.value = true
}

function openEdit(s) {
  editingName.value = s.name
  form.value = { ...s, password: '' }   // never prefill password
  formError.value = ''
  showForm.value = true
}

function closeForm() {
  // Wipe password from in-memory form on close
  form.value.password = ''
  showForm.value = false
}

async function submitForm() {
  busy.value = true
  formError.value = ''
  try {
    if (editingName.value) {
      const patch = { ...form.value }
      delete patch.name
      // Empty password means "don't change"
      if (!patch.password) delete patch.password
      await put(`/api/publish-servers/${encodeURIComponent(editingName.value)}`, patch)
    } else {
      await post('/api/publish-servers', { ...form.value })
    }
    form.value.password = ''
    showForm.value = false
    await load()
  } catch (e) {
    formError.value = (e?.message || String(e)).slice(0, 200)
  } finally {
    busy.value = false
  }
}

function confirmDelete(s) {
  pendingDelete.value = s
}

async function doDelete() {
  const name = pendingDelete.value?.name
  pendingDelete.value = null
  if (!name) return
  try {
    await del(`/api/publish-servers/${encodeURIComponent(name)}`)
    await load()
  } catch (e) {
    formError.value = (e?.message || String(e)).slice(0, 200)
  }
}

onMounted(load)
</script>

<style scoped>
.page { padding: 24px; max-width: 900px; margin: 0 auto; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.empty { color: #888; padding: 24px; text-align: center; }
.server-table { width: 100%; border-collapse: collapse; }
.server-table th, .server-table td { padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }
.server-table th { background: #fafafa; font-weight: 500; font-size: 13px; }
.server-table td code { font-family: ui-monospace, monospace; font-size: 12px; color: #555; }
.actions { display: flex; gap: 6px; }
.form-error { color: #d03050; font-size: 13px; margin-top: 8px; }
</style>
