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
            <el-menu-item index="/model-select">
              <el-icon><Grid /></el-icon>
              <span>模型选择</span>
            </el-menu-item>
            <el-menu-item index="/tasks">
              <el-icon><Operation /></el-icon>
              <span>任务执行</span>
            </el-menu-item>
            <el-menu-item index="/results">
              <el-icon><DataLine /></el-icon>
              <span>结果查看</span>
            </el-menu-item>
            <el-menu-item index="/settings">
              <el-icon><Setting /></el-icon>
              <span>系统设置</span>
            </el-menu-item>
          </el-menu>
        </el-aside>
        <el-main class="app-main">
          <router-view />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const activeMenu = computed(() => route.path)
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
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.main-layout {
  height: 100%;
}

.app-header {
  background: rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  padding: 0 20px;
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
  background: rgba(255, 255, 255, 0.03);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
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
  background: rgba(64, 158, 255, 0.2);
  color: #409eff;
}

.app-main {
  background: rgba(255, 255, 255, 0.02);
  padding: 20px;
  overflow-y: auto;
}
</style>