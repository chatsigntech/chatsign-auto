export default {
  app: {
    title: 'ChatSign Orchestrator',
    subtitle: '6-Phase Sign Language Pipeline'
  },
  nav: {
    dashboard: 'Dashboard',
    logout: 'Logout'
  },
  login: {
    title: 'Sign In',
    username: 'Username',
    password: 'Password',
    submit: 'Sign In',
    error: 'Invalid username or password'
  },
  dashboard: {
    title: 'Pipeline Tasks',
    create: 'New Task',
    empty: 'No tasks yet. Create one to get started.',
    filterAll: 'All',
    filterPending: 'Pending',
    filterRunning: 'Running',
    filterCompleted: 'Completed',
    filterFailed: 'Failed',
    filterPaused: 'Paused'
  },
  task: {
    name: 'Task Name',
    namePlaceholder: 'Enter task name',
    batchPlaceholder: 'Batch name filter (optional, e.g. school_unmatch)',
    create: 'Create',
    cancel: 'Cancel',
    run: 'Run',
    pause: 'Pause',
    resume: 'Resume',
    delete: 'Delete',
    confirmDelete: 'Are you sure you want to delete this task?',
    phase: 'Phase',
    currentPhase: 'Current Phase',
    createdAt: 'Created',
    updatedAt: 'Updated',
    status: 'Status',
    progress: 'Progress',
    errorMessage: 'Error',
    gpuId: 'GPU'
  },
  status: {
    pending: 'Pending',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
    paused: 'Paused'
  },
  accuracy: {
    title: 'Video Collection',
    allBatches: 'All',
    totalSubmissions: 'Total Submissions',
    approved: 'Approved',
    rejected: 'Rejected',
    pendingReview: 'Pending Review',
    readyMsg: '{count} approved videos ready for pipeline',
    notReadyMsg: 'No approved videos yet. Collect and review videos first.',
    noData: 'No collection data found',
    sentence: 'Sentence',
    translator: 'Translator',
    filename: 'Filename'
  },
  augConfig: {
    title: 'Augmentation Configuration',
    subtitle: 'Configure data augmentation parameters for Phase 7',
    summary: '{count} augmentations enabled per input video',
    save: 'Save Configuration',
    reset: 'Reset to Defaults',
    saveSuccess: 'Configuration saved successfully',
    saveError: 'Failed to save configuration',
    sections: {
      cv2d: '2D CV Augmentation',
      temporal: 'Temporal Augmentation',
      view3d: '3D View Augmentation',
      identity: 'Identity Cross-Reenactment'
    },
    categories: {
      crop: 'Crop',
      rotate: 'Rotate',
      perspective: 'Perspective',
      brightness: 'Brightness',
      contrast: 'Contrast',
      saturation: 'Saturation',
      grayscale: 'Grayscale',
      hue: 'Hue Shift',
      gamma: 'Gamma',
      jitter: 'Color Jitter',
      speed: 'Speed Change',
      subsample: 'Frame Subsample',
      yaw: 'Horizontal Rotation (Yaw)',
      pitch: 'Vertical Rotation (Pitch)',
      zoom: 'Zoom',
      combined: 'Combined Viewpoints'
    }
  },
  phases: {
    1: 'Video Collection',
    2: 'Pseudo-gloss Extraction',
    3: 'Annotation Organization',
    4: 'Person Transfer',
    5: 'Video Processing',
    6: 'Frame Interpolation',
    7: 'Data Augmentation',
    8: 'Model Training'
  }
}
