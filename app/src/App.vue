<template>
  <div class="app-container">
    <el-container class="main-layout">
      <el-header class="app-header">
        <div class="header-content">
          <div class="logo">
            <el-icon size="24"><Monitor /></el-icon>
            <span class="title">航空发动机状态监控系统</span>
          </div>
          <div class="header-info">
            <el-tag :type="dataStore.hasData ? 'success' : 'info'" effect="plain">
              {{ dataStore.hasData ? '数据已挂载' : '等待导入' }}
            </el-tag>
            <el-tag type="success" effect="dark">本地运行</el-tag>
            <el-tag type="info">数据安全</el-tag>
          </div>
        </div>
      </el-header>
      <el-container>
        <el-aside width="220px" class="app-aside">
          <el-menu
            :default-active="activeMenu"
            router
            class="side-menu"
          >
            <el-menu-item index="/data-import">
              <el-icon><Upload /></el-icon>
              <span>数据导入</span>
            </el-menu-item>
            <el-menu-item index="/tasks">
              <el-icon><Operation /></el-icon>
              <span>分析任务</span>
            </el-menu-item>
            <el-menu-item index="/model-select">
              <el-icon><Grid /></el-icon>
              <span>模型配置</span>
            </el-menu-item>
            <el-menu-item index="/results">
              <el-icon><DataLine /></el-icon>
              <span>分析结果</span>
            </el-menu-item>
            <el-menu-item index="/settings">
              <el-icon><Setting /></el-icon>
              <span>系统设置</span>
            </el-menu-item>
          </el-menu>
        </el-aside>
        <el-main class="app-main">
          <div v-if="route.path !== '/settings'" class="workflow-strip">
            <el-steps :active="activeStep" finish-status="success" align-center>
              <el-step title="导入数据" />
              <el-step title="选择任务" />
              <el-step title="配置模型" />
              <el-step title="查看结果" />
            </el-steps>
          </div>
          <router-view />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useDataStore } from './stores/data'

const route = useRoute()
const dataStore = useDataStore()
const activeMenu = computed(() => route.path)
const activeStep = computed(() => {
  const order: Record<string, number> = {
    '/data-import': 0,
    '/tasks': 1,
    '/model-select': 2,
    '/results': 3
  }
  return order[route.path] ?? 0
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, #app, .app-container {
  height: 100%;
  width: 100%;
}

.app-container {
  background: #eef2f7;
  color: #1f2937;
}

.main-layout {
  height: 100%;
}

.app-header {
  background: #0f172a;
  border-bottom: 1px solid #1e293b;
  display: flex;
  align-items: center;
  padding: 0 20px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.18);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #fff;
}

.title {
  font-size: 18px;
  font-weight: 600;
}

.header-info {
  display: flex;
  gap: 10px;
}

.app-aside {
  background: #111827;
  border-right: 1px solid #1f2937;
}

.side-menu {
  border-right: none;
  background: transparent;
  height: 100%;
}

.side-menu .el-menu-item {
  color: rgba(255, 255, 255, 0.7);
}

.side-menu .el-menu-item:hover {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.side-menu .el-menu-item.is-active {
  background: rgba(59, 130, 246, 0.16);
  color: #93c5fd;
  border-right: 3px solid #3b82f6;
}

.app-main {
  background: #f3f6fb;
  padding: 18px 22px 28px;
  overflow-y: auto;
}

.workflow-strip {
  max-width: 1180px;
  margin: 0 auto 16px;
  padding: 14px 18px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
</style>
