<script setup>
import { darkTheme, zhCN, enUS } from 'naive-ui'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { locale } = useI18n()

const naiveLocale = computed(() => locale.value === 'zh' ? zhCN : enUS)

const themeOverrides = {
  common: {
    primaryColor: '#00CFC8',
    primaryColorHover: '#00E5DD',
    primaryColorPressed: '#00B3AD',
    bodyColor: '#0A0A0F',
    cardColor: '#1A1A2E',
    modalColor: '#1A1A2E',
    popoverColor: '#1A1A2E',
    inputColor: '#1A1A2E',
    tableColor: '#1A1A2E',
    textColorBase: '#E2E8F0',
    textColor1: '#E2E8F0',
    textColor2: 'rgba(226, 232, 240, 0.82)',
    textColor3: 'rgba(226, 232, 240, 0.52)',
    borderColor: 'rgba(0, 207, 200, 0.15)',
    dividerColor: 'rgba(0, 207, 200, 0.1)',
    hoverColor: '#252540',
    tagColor: '#252540'
  },
  Button: {
    textColorPrimary: '#0A0A0F',
    textColorHoverPrimary: '#0A0A0F',
    textColorPressedPrimary: '#0A0A0F'
  },
  Card: {
    borderColor: 'rgba(0, 207, 200, 0.15)',
    colorModal: '#1A1A2E'
  },
  Input: {
    borderFocus: '#00CFC8',
    borderHover: 'rgba(0, 207, 200, 0.4)'
  },
  Tag: {
    borderPrimary: 'rgba(0, 207, 200, 0.3)'
  },
  Progress: {
    railColor: 'rgba(0, 207, 200, 0.1)'
  }
}
</script>

<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides" :locale="naiveLocale">
    <n-message-provider>
      <n-dialog-provider>
        <router-view />
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>
