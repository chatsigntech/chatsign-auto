<script setup>
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'

const props = defineProps({ show: Boolean })
const emit = defineEmits(['update:show', 'selected'])

const { t } = useI18n()
const { get } = useApi()

const videos = ref([])
const loading = ref(false)
const search = ref('')

async function fetchVideos() {
  loading.value = true
  try {
    const data = await get('/api/phase3-test/videos')
    videos.value = data.videos.filter(v => v.exists)
  } catch {
    // handled by useApi
  } finally {
    loading.value = false
  }
}

watch(() => props.show, (val) => {
  if (val) fetchVideos()
})

const filtered = computed(() => {
  if (!search.value) return videos.value
  const q = search.value.toLowerCase()
  return videos.value.filter(v =>
    (v.sentence_text || '').toLowerCase().includes(q) ||
    (v.filename || '').toLowerCase().includes(q) ||
    (v.translator_id || '').toLowerCase().includes(q)
  )
})

function handleSelect(videoId) {
  emit('selected', videoId)
  emit('update:show', false)
}

function handleClose() {
  emit('update:show', false)
}
</script>

<template>
  <n-modal :show="props.show" @update:show="handleClose" preset="card"
    :title="t('phase3Test.selectVideo')" style="max-width: 800px; max-height: 85vh;">
    <n-input v-model:value="search" :placeholder="t('phase3Test.searchPlaceholder')"
      clearable style="margin-bottom: 12px;" />
    <n-spin :show="loading">
      <div style="max-height: 500px; overflow: auto;">
        <n-table :single-line="false" size="small" striped>
          <thead>
            <tr>
              <th style="width: 200px;">{{ t('phase3Test.sentence') }}</th>
              <th style="width: 150px;">{{ t('phase3Test.filename') }}</th>
              <th style="width: 90px;">{{ t('phase3Test.translator') }}</th>
              <th style="width: 70px;">{{ t('phase3Test.source') }}</th>
              <th style="width: 100px;"></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="v in filtered" :key="v.video_id">
              <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                :title="v.sentence_text">{{ v.sentence_text }}</td>
              <td style="max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                :title="v.filename">{{ v.filename }}</td>
              <td>{{ v.translator_id }}</td>
              <td>{{ v.source }}</td>
              <td>
                <n-space :size="4">
                  <n-button size="tiny" type="primary" @click="handleSelect(v.video_id)">
                    {{ t('phase3Test.runPhase3') }}
                  </n-button>
                </n-space>
              </td>
            </tr>
          </tbody>
        </n-table>
      </div>
    </n-spin>
  </n-modal>
</template>
