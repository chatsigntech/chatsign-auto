import { ref, computed } from 'vue'
import { useApi } from '../composables/useApi.js'
import { useAuth } from '../composables/useAuth.js'

const CAPTURE_FPS = 15
const CAPTURE_INTERVAL = Math.round(1000 / CAPTURE_FPS)
const JPEG_QUALITY = 0.7
const MAX_BUFFERED_BYTES = 256 * 1024  // skip frames when WS buffer exceeds this

export function useRecognition() {
  const { get } = useApi()
  const { token } = useAuth()

  const isConnected = ref(false)
  const isStreaming = ref(false)
  const isModelLoading = ref(false)
  const selectedModel = ref(null)
  const models = ref([])
  const lastPrediction = ref(null)   // raw server message
  const error = ref(null)

  const results = computed(() => {
    const msg = lastPrediction.value
    if (!msg || !msg.tokens) return []
    return msg.tokens.map((t, i) => ({
      token: t,
      score: i === msg.tokens.length - 1 ? msg.latest_score : null,
    }))
  })

  const sentence = computed(() => lastPrediction.value?.sentence || '')

  const stats = computed(() => ({
    pose_frames: lastPrediction.value?.pose_frames || 0,
    windows: lastPrediction.value?.windows || 0,
  }))

  const modelOptions = computed(() =>
    models.value.map(m => ({
      label: `${m.task_name} (${m.vocab_size} glosses)`,
      value: m.task_id,
    }))
  )

  let ws = null
  let captureTimer = null
  let mediaStream = null
  let canvas = null

  async function loadModels() {
    try {
      error.value = null
      models.value = await get('/api/recognition/models')
    } catch (e) {
      error.value = e.message
      models.value = []
    }
  }

  function _clearResults() {
    lastPrediction.value = null
  }

  async function startSession(videoEl) {
    if (!selectedModel.value) {
      error.value = 'Please select a model'
      return
    }
    error.value = null

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      })
      videoEl.srcObject = mediaStream
      await videoEl.play()
    } catch (e) {
      error.value = `Camera access denied: ${e.message}`
      return
    }

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${proto}//${location.host}/api/recognition/ws/${selectedModel.value}?token=${encodeURIComponent(token.value)}`

    isModelLoading.value = true
    ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      isConnected.value = true
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)

        if (msg.type === 'ready') {
          isModelLoading.value = false
          isStreaming.value = true
          _startCapture(videoEl)
        } else if (msg.type === 'prediction') {
          lastPrediction.value = msg
        } else if (msg.type === 'reset_ack') {
          _clearResults()
        } else if (msg.type === 'error') {
          error.value = msg.message
          isModelLoading.value = false
        }
      } catch {
        // ignore parse errors
      }
    }

    ws.onclose = () => {
      isConnected.value = false
      isStreaming.value = false
      isModelLoading.value = false
      _stopCapture()
    }

    ws.onerror = () => {
      error.value = 'WebSocket connection error'
      isModelLoading.value = false
    }
  }

  function stopSession() {
    _stopCapture()

    if (ws) {
      ws.close()
      ws = null
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop())
      mediaStream = null
    }

    isConnected.value = false
    isStreaming.value = false
    isModelLoading.value = false
  }

  function resetSession() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'reset' }))
    }
    _clearResults()
  }

  function _startCapture(videoEl) {
    if (captureTimer) return

    canvas = document.createElement('canvas')
    canvas.width = 640
    canvas.height = 480
    const ctx = canvas.getContext('2d')

    captureTimer = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      if (videoEl.readyState < 2) return
      // Backpressure: skip frame if WS send buffer is full
      if (ws.bufferedAmount > MAX_BUFFERED_BYTES) return

      ctx.drawImage(videoEl, 0, 0, 640, 480)
      canvas.toBlob(
        (blob) => {
          if (blob && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(blob)
          }
        },
        'image/jpeg',
        JPEG_QUALITY,
      )
    }, CAPTURE_INTERVAL)
  }

  function _stopCapture() {
    if (captureTimer) {
      clearInterval(captureTimer)
      captureTimer = null
    }
    if (canvas) {
      canvas.width = 0
      canvas.height = 0
      canvas = null
    }
  }

  return {
    isConnected,
    isStreaming,
    isModelLoading,
    selectedModel,
    models,
    modelOptions,
    results,
    sentence,
    stats,
    error,
    loadModels,
    startSession,
    stopSession,
    resetSession,
  }
}
