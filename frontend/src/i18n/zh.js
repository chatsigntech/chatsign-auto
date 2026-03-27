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
    preset: '增广预设',
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
  phases: {
    1: '视频采集',
    2: '伪注解提取',
    3: '标注整理',
    4: '视频预处理',
    5: '数据增广',
    6: '模型训练'
  }
}
