<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useApi } from '../composables/useApi.js'
import { useMessage } from 'naive-ui'
import AppHeader from '../components/AppHeader.vue'
import {
  SaveOutline,
  RefreshOutline,
  ColorPaletteOutline,
  TimerOutline,
  CubeOutline,
  PeopleOutline
} from '@vicons/ionicons5'

const { t } = useI18n()
const { get, put } = useApi()
const message = useMessage()

const config = ref(null)
const loading = ref(false)
const saving = ref(false)

// ---- Computed helpers ----

const enabledCount = computed(() => {
  if (!config.value) return 0
  let count = 0
  // cv2d
  if (config.value.cv2d) {
    for (const augs of Object.values(config.value.cv2d)) {
      if (Array.isArray(augs)) {
        count += augs.filter(a => a.enabled).length
      }
    }
  }
  // temporal
  if (config.value.temporal) {
    for (const augs of Object.values(config.value.temporal)) {
      if (Array.isArray(augs)) {
        count += augs.filter(a => a.enabled).length
      } else if (typeof augs === 'object' && augs.enabled !== undefined) {
        if (augs.enabled) count++
      }
    }
  }
  // view3d
  if (config.value.view3d) {
    for (const augs of Object.values(config.value.view3d)) {
      if (Array.isArray(augs)) {
        count += augs.filter(a => a.enabled).length
      } else if (typeof augs === 'object' && augs.enabled !== undefined) {
        if (augs.enabled) count++
      }
    }
  }
  // identity
  if (config.value.identity) {
    for (const augs of Object.values(config.value.identity)) {
      if (Array.isArray(augs)) {
        count += augs.filter(a => a.enabled).length
      } else if (typeof augs === 'object' && augs.enabled !== undefined) {
        if (augs.enabled) count++
      }
    }
  }
  return count
})

function sectionCount(section) {
  if (!config.value || !config.value[section]) return 0
  let count = 0
  for (const augs of Object.values(config.value[section])) {
    if (Array.isArray(augs)) {
      count += augs.filter(a => a.enabled).length
    } else if (typeof augs === 'object' && augs.enabled !== undefined) {
      if (augs.enabled) count++
    }
  }
  return count
}

function sectionTotal(section) {
  if (!config.value || !config.value[section]) return 0
  let count = 0
  for (const augs of Object.values(config.value[section])) {
    if (Array.isArray(augs)) {
      count += augs.length
    } else if (typeof augs === 'object' && augs.enabled !== undefined) {
      count++
    }
  }
  return count
}

function toggleSection(section, val) {
  if (!config.value || !config.value[section]) return
  for (const augs of Object.values(config.value[section])) {
    if (Array.isArray(augs)) {
      augs.forEach(a => { a.enabled = val })
    } else if (typeof augs === 'object' && augs.enabled !== undefined) {
      augs.enabled = val
    }
  }
}

function isSectionEnabled(section) {
  return sectionCount(section) > 0
}

function formatParams(aug) {
  const params = []
  for (const [k, v] of Object.entries(aug)) {
    if (k === 'enabled' || k === 'name' || k === 'label' || k === 'type' || k === 'category') continue
    if (typeof v === 'object' && v !== null) {
      params.push(`${k}: ${JSON.stringify(v)}`)
    } else {
      params.push(`${k}: ${v}`)
    }
  }
  return params
}

function getCategoryLabel(key) {
  const k = `augConfig.categories.${key}`
  const translated = t(k)
  // If translation returns the key itself, just capitalize
  return translated === k ? key.charAt(0).toUpperCase() + key.slice(1) : translated
}

function getGroupedItems(sectionData) {
  // Group items by category if they have one, otherwise by key
  const groups = {}
  for (const [key, items] of Object.entries(sectionData)) {
    if (Array.isArray(items)) {
      groups[key] = items
    } else if (typeof items === 'object' && items.enabled !== undefined) {
      // Single item, wrap in array
      if (!groups['_singles']) groups['_singles'] = []
      groups['_singles'].push({ ...items, _key: key })
    }
  }
  return groups
}

// ---- API ----

async function fetchConfig() {
  loading.value = true
  try {
    config.value = await get('/api/config/augmentation')
  } catch (e) {
    config.value = null
  } finally {
    loading.value = false
  }
}

async function saveConfig() {
  saving.value = true
  try {
    await put('/api/config/augmentation', config.value)
    message.success(t('augConfig.saveSuccess'))
  } catch (e) {
    message.error(t('augConfig.saveError'))
  } finally {
    saving.value = false
  }
}

async function resetConfig() {
  await fetchConfig()
}

onMounted(() => {
  fetchConfig()
})
</script>

<template>
  <div class="aug-page">
    <AppHeader />
    <div class="aug-content">
      <h2 class="page-title">{{ t('augConfig.title') }}</h2>
      <p class="page-subtitle">{{ t('augConfig.subtitle') }}</p>

      <n-spin :show="loading">
        <template v-if="config">
          <!-- Summary bar -->
          <n-card class="summary-bar" size="small">
            <n-space align="center" justify="space-between">
              <n-statistic :value="enabledCount">
                <template #label>{{ t('augConfig.summary', { count: enabledCount }) }}</template>
              </n-statistic>
              <n-space>
                <n-button @click="resetConfig" :disabled="saving">
                  <template #icon><n-icon :component="RefreshOutline" /></template>
                  {{ t('augConfig.reset') }}
                </n-button>
                <n-button type="primary" @click="saveConfig" :loading="saving">
                  <template #icon><n-icon :component="SaveOutline" /></template>
                  {{ t('augConfig.save') }}
                </n-button>
              </n-space>
            </n-space>
          </n-card>

          <!-- Sections -->
          <n-collapse default-expanded-names="cv2d" style="margin-top: 20px;">

            <!-- Section 1: 2D CV Augmentation -->
            <n-collapse-item v-if="config.cv2d" name="cv2d">
              <template #header>
                <n-space align="center" :size="12">
                  <n-icon :component="ColorPaletteOutline" :size="20" color="#00CFC8" />
                  <span class="section-title">{{ t('augConfig.sections.cv2d') }}</span>
                  <n-tag size="small" :type="sectionCount('cv2d') > 0 ? 'success' : 'default'" round>
                    {{ sectionCount('cv2d') }} / {{ sectionTotal('cv2d') }}
                  </n-tag>
                </n-space>
              </template>
              <template #header-extra>
                <n-switch
                  :value="isSectionEnabled('cv2d')"
                  @update:value="v => toggleSection('cv2d', v)"
                  @click.stop
                />
              </template>
              <div v-for="(items, category) in getGroupedItems(config.cv2d)" :key="category" class="category-group">
                <div class="category-label">{{ getCategoryLabel(category) }}</div>
                <n-card v-for="(aug, idx) in items" :key="idx" size="small" class="aug-item">
                  <n-space align="center" justify="space-between">
                    <n-space align="center" :size="12">
                      <n-switch v-model:value="aug.enabled" size="small" />
                      <span class="aug-name">{{ aug.name || aug.label || aug._key || `${category} #${idx + 1}` }}</span>
                    </n-space>
                    <n-space :size="6">
                      <n-tag v-for="p in formatParams(aug)" :key="p" size="tiny" :bordered="false" type="info">
                        {{ p }}
                      </n-tag>
                    </n-space>
                  </n-space>
                </n-card>
              </div>
            </n-collapse-item>

            <!-- Section 2: Temporal Augmentation -->
            <n-collapse-item v-if="config.temporal" name="temporal">
              <template #header>
                <n-space align="center" :size="12">
                  <n-icon :component="TimerOutline" :size="20" color="#F0A020" />
                  <span class="section-title">{{ t('augConfig.sections.temporal') }}</span>
                  <n-tag size="small" :type="sectionCount('temporal') > 0 ? 'success' : 'default'" round>
                    {{ sectionCount('temporal') }} / {{ sectionTotal('temporal') }}
                  </n-tag>
                </n-space>
              </template>
              <template #header-extra>
                <n-switch
                  :value="isSectionEnabled('temporal')"
                  @update:value="v => toggleSection('temporal', v)"
                  @click.stop
                />
              </template>
              <div v-for="(items, category) in getGroupedItems(config.temporal)" :key="category" class="category-group">
                <div class="category-label">{{ getCategoryLabel(category) }}</div>
                <n-card v-for="(aug, idx) in items" :key="idx" size="small" class="aug-item">
                  <n-space align="center" justify="space-between">
                    <n-space align="center" :size="12">
                      <n-switch v-model:value="aug.enabled" size="small" />
                      <span class="aug-name">{{ aug.name || aug.label || aug._key || `${category} #${idx + 1}` }}</span>
                    </n-space>
                    <n-space :size="6">
                      <n-tag v-for="p in formatParams(aug)" :key="p" size="tiny" :bordered="false" type="info">
                        {{ p }}
                      </n-tag>
                    </n-space>
                  </n-space>
                </n-card>
              </div>
            </n-collapse-item>

            <!-- Section 3: 3D View Augmentation -->
            <n-collapse-item v-if="config.view3d" name="view3d">
              <template #header>
                <n-space align="center" :size="12">
                  <n-icon :component="CubeOutline" :size="20" color="#A855F7" />
                  <span class="section-title">{{ t('augConfig.sections.view3d') }}</span>
                  <n-tag size="small" :type="sectionCount('view3d') > 0 ? 'success' : 'default'" round>
                    {{ sectionCount('view3d') }} / {{ sectionTotal('view3d') }}
                  </n-tag>
                </n-space>
              </template>
              <template #header-extra>
                <n-switch
                  :value="isSectionEnabled('view3d')"
                  @update:value="v => toggleSection('view3d', v)"
                  @click.stop
                />
              </template>
              <div v-for="(items, category) in getGroupedItems(config.view3d)" :key="category" class="category-group">
                <div class="category-label">{{ getCategoryLabel(category) }}</div>
                <n-card v-for="(aug, idx) in items" :key="idx" size="small" class="aug-item">
                  <n-space align="center" justify="space-between">
                    <n-space align="center" :size="12">
                      <n-switch v-model:value="aug.enabled" size="small" />
                      <span class="aug-name">{{ aug.name || aug.label || aug._key || `${category} #${idx + 1}` }}</span>
                    </n-space>
                    <n-space :size="6">
                      <n-tag v-for="p in formatParams(aug)" :key="p" size="tiny" :bordered="false" type="info">
                        {{ p }}
                      </n-tag>
                    </n-space>
                  </n-space>
                </n-card>
              </div>
            </n-collapse-item>

            <!-- Section 4: Identity Cross-Reenactment -->
            <n-collapse-item v-if="config.identity" name="identity">
              <template #header>
                <n-space align="center" :size="12">
                  <n-icon :component="PeopleOutline" :size="20" color="#E74C3C" />
                  <span class="section-title">{{ t('augConfig.sections.identity') }}</span>
                  <n-tag size="small" :type="sectionCount('identity') > 0 ? 'success' : 'default'" round>
                    {{ sectionCount('identity') }} / {{ sectionTotal('identity') }}
                  </n-tag>
                </n-space>
              </template>
              <template #header-extra>
                <n-switch
                  :value="isSectionEnabled('identity')"
                  @update:value="v => toggleSection('identity', v)"
                  @click.stop
                />
              </template>
              <div v-for="(items, category) in getGroupedItems(config.identity)" :key="category" class="category-group">
                <div class="category-label">{{ getCategoryLabel(category) }}</div>
                <n-card v-for="(aug, idx) in items" :key="idx" size="small" class="aug-item">
                  <n-space align="center" justify="space-between">
                    <n-space align="center" :size="12">
                      <n-switch v-model:value="aug.enabled" size="small" />
                      <span class="aug-name">{{ aug.name || aug.label || aug._key || `${category} #${idx + 1}` }}</span>
                    </n-space>
                    <n-space :size="6">
                      <n-tag v-for="p in formatParams(aug)" :key="p" size="tiny" :bordered="false" type="info">
                        {{ p }}
                      </n-tag>
                    </n-space>
                  </n-space>
                </n-card>
              </div>
            </n-collapse-item>

          </n-collapse>

          <!-- Bottom action bar -->
          <div class="bottom-bar">
            <n-space justify="end">
              <n-button @click="resetConfig" :disabled="saving">
                <template #icon><n-icon :component="RefreshOutline" /></template>
                {{ t('augConfig.reset') }}
              </n-button>
              <n-button type="primary" @click="saveConfig" :loading="saving">
                <template #icon><n-icon :component="SaveOutline" /></template>
                {{ t('augConfig.save') }}
              </n-button>
            </n-space>
          </div>
        </template>

        <n-empty v-else-if="!loading" description="No augmentation config found" style="margin-top: 80px;" />
      </n-spin>
    </div>
  </div>
</template>

<style scoped>
.aug-page {
  min-height: 100vh;
}
.aug-content {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px;
}
.page-title {
  font-size: 22px;
  font-weight: 600;
  margin-bottom: 4px;
}
.page-subtitle {
  font-size: 14px;
  color: rgba(226, 232, 240, 0.6);
  margin-bottom: 20px;
}
.summary-bar {
  margin-bottom: 4px;
}
.section-title {
  font-size: 16px;
  font-weight: 600;
}
.category-group {
  margin-bottom: 16px;
}
.category-label {
  font-size: 13px;
  font-weight: 600;
  color: rgba(226, 232, 240, 0.5);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
  margin-top: 8px;
}
.aug-item {
  margin-bottom: 6px;
}
.aug-name {
  font-size: 14px;
  color: #E2E8F0;
}
.bottom-bar {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid rgba(0, 207, 200, 0.1);
}
</style>
