<template>
  <div class="settings-page">
    <el-card class="settings-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span>系统设置</span>
        </div>
      </template>

      <el-tabs>
        <el-tab-pane label="默认参数">
          <el-form label-width="150px" class="settings-form">
            <el-form-item label="默认预测步数">
              <el-input-number v-model="store.appSettings.defaultForecastLength" :min="1" :max="10000" />
            </el-form-item>

            <el-form-item label="默认异常阈值">
              <el-slider v-model="store.appSettings.defaultAnomalyThreshold" :min="0" :max="10" :step="0.1" show-input />
            </el-form-item>

            <el-form-item label="默认修复方法">
              <el-select v-model="store.appSettings.defaultRepairMethod">
                <el-option label="多任务模型重构" value="model_reconstruction" />
                <el-option label="线性插值兜底" value="linear" />
                <el-option label="中位数填充兜底" value="median" />
              </el-select>
            </el-form-item>

            <el-form-item label="默认上下文窗口">
              <el-input-number v-model="store.appSettings.defaultContextLength" :min="16" :max="2048" />
            </el-form-item>

            <el-form-item label="数据预览行数">
              <el-input-number v-model="store.appSettings.previewRows" :min="10" :max="500" />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="环境与模型">
          <el-form label-width="150px" class="settings-form">
            <el-form-item label="Python 解释器">
              <el-input v-model="store.appSettings.pythonPath" placeholder="python" />
            </el-form-item>

            <el-form-item label="模型目录">
              <el-input v-model="store.appSettings.modelDir" placeholder="weights" />
            </el-form-item>

            <el-form-item label="CPU 动态量化">
              <el-switch v-model="store.appSettings.enableCpuQuantization" />
            </el-form-item>

            <el-form-item label="自动保存结果">
              <el-switch v-model="store.appSettings.autoSaveResults" />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="关于与安全">
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
            <p>2. 使用 Tauri 框架，应用完全运行在您的设备上</p>
            <p>3. 模型推理在本地执行，保护您的数据隐私</p>
          </el-alert>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { useDataStore } from '../stores/data'

const store = useDataStore()
</script>

<style scoped>
.settings-page {
  height: 100%;
  max-width: 1000px;
  margin: 0 auto;
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

.settings-form {
  max-width: 760px;
  padding-top: 8px;
}
</style>
