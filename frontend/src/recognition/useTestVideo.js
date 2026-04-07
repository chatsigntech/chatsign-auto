import { ref, computed, onUnmounted } from 'vue'
import { useApi } from '../composables/useApi.js'

const POLL_INTERVAL = 2000

export function useTestVideo() {
  const { get, post } = useApi()

  const jobId = ref(null)
  const jobStatus = ref(null)   // pending | generating | completed | failed
  const progress = ref(0)
  const sentences = ref([])      // [{index, sentence_text, start_time, end_time, aug_name}]
  const videoUrl = ref(null)
  const fps = ref(0)
  const duration = ref(0)
  const error = ref(null)
  const isGenerating = computed(() =>
    jobStatus.value === 'pending' || jobStatus.value === 'generating'
  )

  // Playback time sync
  const currentTime = ref(0)

  const currentSentenceIndex = computed(() => {
    const t = currentTime.value
    return sentences.value.findIndex(
      s => t >= s.start_time && t < s.end_time
    )
  })

  let pollTimer = null
  let _lastSentenceIdx = -1

  async function generate(taskId) {
    error.value = null
    jobStatus.value = 'pending'
    progress.value = 0
    sentences.value = []
    videoUrl.value = null
    _lastSentenceIdx = -1

    try {
      const res = await post(`/api/test-video/generate/${taskId}`)
      jobId.value = res.job_id
      _startPolling()
    } catch (e) {
      error.value = e.message
      jobStatus.value = 'failed'
    }
  }

  function _startPolling() {
    _stopPolling()
    pollTimer = setInterval(async () => {
      try {
        const res = await get(`/api/test-video/status/${jobId.value}`)
        jobStatus.value = res.status
        progress.value = res.progress

        if (res.status === 'completed') {
          videoUrl.value = res.video_url
          sentences.value = res.sentences
          fps.value = res.fps
          duration.value = res.duration
          _stopPolling()
        } else if (res.status === 'failed') {
          error.value = res.error || 'Generation failed'
          _stopPolling()
        }
      } catch (e) {
        error.value = e.message
        _stopPolling()
      }
    }, POLL_INTERVAL)
  }

  function _stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  onUnmounted(_stopPolling)

  function onTimeUpdate(time) {
    currentTime.value = time
  }

  /**
   * Check if we just crossed a sentence boundary and need to reset.
   * Returns true if a reset should be sent.
   */
  function checkBoundary() {
    const idx = currentSentenceIndex.value
    if (idx >= 0 && idx !== _lastSentenceIdx && _lastSentenceIdx >= 0) {
      _lastSentenceIdx = idx
      return true
    }
    _lastSentenceIdx = idx
    return false
  }

  function reset() {
    _stopPolling()
    jobId.value = null
    jobStatus.value = null
    progress.value = 0
    sentences.value = []
    videoUrl.value = null
    error.value = null
    currentTime.value = 0
    _lastSentenceIdx = -1
  }

  return {
    jobId,
    jobStatus,
    progress,
    sentences,
    videoUrl,
    fps,
    duration,
    error,
    isGenerating,
    currentTime,
    currentSentenceIndex,
    generate,
    onTimeUpdate,
    checkBoundary,
    reset,
  }
}
