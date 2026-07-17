<template>
  <div class="settings-page">
    <el-card class="settings-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span>系统设置</span>
        </div>
      </template>

      <el-form label-width="150px">
        <el-form-item label="默认预测步数">
          <el-input-number v-model="settings.defaultForecastLength" :min="1" :max="100" />
        </el-form-item>

        <el-form-item label="默认异常阈值">
          <el-slider v-model="settings.defaultAnomalyThreshold" :min="0" :max="10" :step="0.1" show-input />
        </el-form-item>

        <el-form-item label="默认修复方法">
          <el-select v-model="settings.defaultRepairMethod">
            <el-option label="均值填充" value="mean" />
            <el-option label="中位数填充" value="median" />
            <el-option label="线性插值" value="linear" />
            <el-option label="前向填充" value="forward" />
          </el-select>
        </el-form-item>

        <el-form-item label="数据预览行数">
          <el-input-number v-model="settings.previewRows" :min="10" :max="500" />
        </el-form-item>

        <el-form-item label="自动保存结果">
          <el-switch v-model="settings.autoSaveResults" />
        </el-form-item>
      </el-form>

      <el-divider />

      <h3>系统信息</h3>
      <el-descriptions :column="1" border>
        <el-descriptions-item label="应用名称">航空发动机状态监控系统</el-descriptions-item>
        <el-descriptions-item label="版本">0.1.0</el-descriptions-item>
        <el-descriptions-item label="运行模式">本地运行（数据安全）</el-descriptions-item>
        <el-descriptions-item label="技术栈">Tauri + Vue3 + Rust</el-descriptions-item>
      </el-descriptions>

      <el-divider />

      <h3>数据安全说明</h3>
      <el-alert type="success" :closable="false">
        <template #title>
          <strong>数据安全承诺</strong>
        </template>
        <p>1. 所有数据处理均在本地完成，数据不会上传到任何服务器</p>
        <p>2. 使用Tauri框架，应用完全运行在您的设备上</p>
        <p>3. 模型推理在本地执行，保护您的数据隐私</p>
      </el-alert>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const settings = ref({
  defaultForecastLength: 24,
  defaultAnomalyThreshold: 3.0,
  defaultRepairMethod: 'mean',
  previewRows: 100,
  autoSaveResults: false
})
</script>

<style scoped>
.settings-page {
  height: 100%;
}

.settings-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 8px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

h3 {
  margin-bottom: 16px;
  color: #303133;
}
</style>