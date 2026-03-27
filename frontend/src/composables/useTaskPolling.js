import { ref, onUnmounted } from 'vue'
import { useApi } from './useApi.js'

export function useTaskPolling(taskId, interval = 3000) {
  const { get } = useApi()
  const task = ref(null)
  const phases = ref([])
  const loading = ref(false)
  let timer = null
  let lastJson = ''

  async function fetchOnce() {
    loading.value = true
    try {
      const data = await get(`/api/tasks/${taskId.value || taskId}`)
      const json = JSON.stringify(data)
      if (json !== lastJson) {
        lastJson = json
        task.value = data.task
        phases.value = data.phases
      }
    } catch (e) {
      // silent on poll errors
    } finally {
      loading.value = false
    }
  }

  function startPolling() {
    stopPolling()
    fetchOnce()
    timer = setInterval(() => {
      const status = task.value?.status
      if (status === 'completed' || status === 'failed') {
        stopPolling()
        return
      }
      fetchOnce()
    }, interval)
  }

  function stopPolling() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  onUnmounted(stopPolling)

  return { task, phases, loading, fetchOnce, startPolling, stopPolling }
}
