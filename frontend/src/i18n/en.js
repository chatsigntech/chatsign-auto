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
    preset: 'Augmentation Preset',
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
