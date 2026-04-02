export default {
  app: {
    title: 'ChatSign Orchestrator',
    subtitle: '6 阶段手语处理流水线'
  },
  nav: {
    dashboard: '控制台',
    logout: '退出登录'
  },
  login: {
    title: '登录',
    username: '用户名',
    password: '密码',
    submit: '登录',
    error: '用户名或密码错误'
  },
  dashboard: {
    title: '流水线任务',
    create: '新建任务',
    empty: '暂无任务，点击新建开始。',
    filterAll: '全部',
    filterPending: '待处理',
    filterRunning: '运行中',
    filterCompleted: '已完成',
    filterFailed: '失败',
    filterPaused: '已暂停'
  },
  task: {
    name: '任务名称',
    namePlaceholder: '请输入任务名称',
    batchPlaceholder: '批次名称过滤（可选，如 school_unmatch）',
    create: '创建',
    cancel: '取消',
    run: '启动',
    pause: '暂停',
    resume: '恢复',
    delete: '删除',
    confirmDelete: '确定要删除这个任务吗？',
    phase: '阶段',
    currentPhase: '当前阶段',
    createdAt: '创建时间',
    updatedAt: '更新时间',
    status: '状态',
    progress: '进度',
    errorMessage: '错误信息',
    gpuId: 'GPU'
  },
  status: {
    pending: '待处理',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    paused: '已暂停'
  },
  accuracy: {
    title: '视频采集',
    allBatches: '全部',
    totalSubmissions: '总提交数',
    approved: '已通过',
    rejected: '已拒绝',
    pendingReview: '待审核',
    readyMsg: '{count} 个已通过视频可用于 Pipeline',
    notReadyMsg: '暂无已通过视频，请先采集并审核手语视频。',
    noData: '未找到采集数据',
    sentence: '句子',
    translator: '录制者',
    filename: '文件名'
  },
  augConfig: {
    title: '增广配置',
    subtitle: '配置 Phase 7 数据增广参数',
    summary: '每个输入视频启用 {count} 种增广',
    save: '保存配置',
    reset: '恢复默认',
    saveSuccess: '配置保存成功',
    saveError: '保存配置失败',
    sections: {
      cv2d: '2D 视觉增广',
      temporal: '时间域增广',
      view3d: '3D 视角增广',
      identity: '身份换人增广'
    },
    categories: {
      crop: '裁剪',
      rotate: '旋转',
      perspective: '透视',
      brightness: '亮度',
      contrast: '对比度',
      saturation: '饱和度',
      grayscale: '灰度',
      hue: '色相偏移',
      gamma: '伽马',
      jitter: '色温',
      speed: '变速',
      subsample: '降采样',
      yaw: '水平旋转',
      pitch: '俯仰旋转',
      zoom: '缩放',
      combined: '组合视角'
    }
  },
  phases: {
    1: '视频采集',
    2: '伪注解提取',
    3: '标注整理',
    4: '换人生成',
    5: '视频处理',
    6: '帧插值合并',
    7: '数据增广',
    8: '模型训练'
  }
}
