<template>
  <div class="model-select-page">
    <el-card class="model-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <h2 class="text-xl font-bold m-0">模型选择与参数配置</h2>
          <el-button text @click="router.push('/tasks')">返回任务选择</el-button>
        </div>
      </template>

      <div v-if="store.hasData && store.currentTask" class="config-layout">
        <div class="status-bar">
          <div>
            <span class="status-label">当前任务</span>
            <strong>{{ getTaskName(store.currentTask) }}</strong>
          </div>
          <div>
            <span class="status-label">当前数据</span>
            <strong>{{ store.currentFile.filename }}</strong>
          </div>
          <div>
            <span class="status-label">数据规模</span>
            <strong>{{ store.currentFile.rows }} 行 × {{ store.currentFile.columns }} 列</strong>
          </div>
        </div>

        <section class="section-panel">
          <div class="section-header">
            <div>
              <h3>分析对象</h3>
              <p>选择本次任务要处理的传感器参数、协变量和数据范围。</p>
            </div>
          </div>
          <el-form label-width="160px" class="param-form">
          
          <el-form-item label="目标分析参数" required>
            <el-select v-model="targetVariables" :multiple="isMultiTarget" :placeholder="isMultiTarget ? '请选择多个参数' : '请选择一个核心参数'" style="width: 100%; max-width: 600px;">
              <el-option v-for="col in store.selectedColumns" :key="col" :label="col" :value="col" />
            </el-select>
          </el-form-item>
          
          <el-form-item label="相关协变量" v-if="store.currentTask === 'covariate_forecast'" required>
            <el-select v-model="covariateVariables" multiple placeholder="请选择影响目标的协变量" style="width: 100%; max-width: 600px;">
              <el-option v-for="col in store.selectedColumns" :key="col" :label="col" :value="col" />
            </el-select>
          </el-form-item>

          <el-form-item label="分析数据范围" v-if="['data_repair', 'anomaly_detection'].includes(store.currentTask)">
            <div style="width: 100%; max-width: 600px;">
              <el-slider v-model="dataRange" range :max="store.currentFile.rows" style="margin-left: 10px; margin-right: 10px;" />
              <div class="el-upload__tip" style="color: #64748b; line-height: 1.4;">
                拖动滑块截取指定的行数区间进行处理。当前截取范围：<strong style="color: #3b82f6;">第 {{ dataRange[0] }} 行</strong> 至 <strong style="color: #3b82f6;">第 {{ dataRange[1] }} 行</strong>，总数据量：{{ store.currentFile.rows }} 行。
              </div>
            </div>
          </el-form-item>

          <el-form-item label="置信区间 (CI)" v-if="['univariate_forecast', 'multivariate_forecast', 'covariate_forecast'].includes(store.currentTask)">
            <el-select v-model="confidenceInterval" style="width: 100%; max-width: 200px;">
              <el-option label="80%" :value="0.80" /><el-option label="90%" :value="0.90" /><el-option label="95%" :value="0.95" />
            </el-select>
          </el-form-item>

          <el-form-item label="可视化选项" v-if="store.currentTask !== 'data_repair'">
            <el-switch v-model="store.showHistoricalData" active-text="在结果图表中展示历史数据" />
          </el-form-item>
        </el-form>
        </section>

        <section class="section-panel">
          <div class="section-header">
            <div>
              <h3>可用模型</h3>
              <p>修复和异常检测只展示支持历史重构的多任务模型。</p>
            </div>
            <el-tag :type="filteredModels.some(model => model.available !== false) ? 'success' : 'warning'" effect="plain">
              {{ filteredModels.some(model => model.available !== false) ? '模型可用' : '待部署' }}
            </el-tag>
          </div>
        <div class="model-grid">
          <div
            v-for="model in filteredModels"
            :key="model.name"
            class="model-card-item"
            :class="{ 'is-active': store.selectedModel === model.name, 'is-disabled': model.available === false }"
            @click="selectModel(model)"
          >
            <div class="model-info">
              <h4>{{ model.name }}</h4><p>{{ model.description }}</p>
              <el-tag v-if="model.available === false" type="warning" size="small">模型未部署</el-tag>
            </div>
            <div class="check-mark" v-if="store.selectedModel === model.name">✓</div>
          </div>
        </div>
        </section>

        <section v-if="store.selectedModel" class="section-panel fade-in">
          <div class="section-header">
            <div>
              <h3>推理参数</h3>
              <p>这些参数会随任务类型变化，并继承系统设置中的默认值。</p>
            </div>
          </div>
          <el-form v-if="displayedParameters.length > 0" label-width="160px" class="param-form">
            <el-form-item v-for="param in displayedParameters" :key="param.name" :label="param.description">
              <div v-if="param.param_type === 'integer'" style="display: flex; align-items: center; gap: 10px;">
                <el-input-number :model-value="Number(param.default)" @update:model-value="(val: number | undefined) => updateParamDefault(param, val)" :min="1" :max="param.name === 'context_length' ? Math.min(store.currentFile.rows, 2048) : 10000" />
                <el-button 
                  v-if="param.name === 'context_length' && ['univariate_forecast', 'multivariate_forecast', 'covariate_forecast'].includes(store.currentTask)" 
                  size="small" type="primary" plain 
                  @click="param.default = String(Math.min(store.currentFile.rows, 2048))"
                >
                  一键最大 (安全视野: 2048步)
                </el-button>
              </div>
              <el-input-number v-else-if="param.param_type === 'float'" :model-value="Number(param.default)" @update:model-value="(val: number | undefined) => updateParamDefault(param, val)" :step="0.1" :min="0.1" :max="10.0" />
            </el-form-item>
          </el-form>
          <el-empty v-else description="当前任务无需额外推理参数" :image-size="72" />
        </section>

        <div class="action-footer mt-8 flex justify-end">
          <el-button type="primary" size="large" @click="executeTask" :disabled="!store.selectedModel || store.loading" :loading="store.loading">执行计算</el-button>
        </div>
      </div>
      <el-empty v-else description="请先完成数据导入并选择分析任务">
        <el-button type="primary" @click="router.push(store.hasData ? '/tasks' : '/data-import')">
          {{ store.hasData ? '去选择任务' : '去导入数据' }}
        </el-button>
      </el-empty>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, watchEffect } from 'vue'
import { useDataStore } from '../stores/data'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

const store = useDataStore()
const router = useRouter()
const isMultiTarget = computed(() => store.currentTask !== 'univariate_forecast')
const targetVariables = ref<string | string[]>(isMultiTarget.value ? [] : '')
const covariateVariables = ref<string[]>([])
const confidenceInterval = ref<number>(0.90)

const dataRange = ref<[number, number]>([0, 0])
watchEffect(() => {
  if (store.currentFile && dataRange.value[1] === 0) {
    dataRange.value = [0, store.currentFile.rows]
  }
})

const getTaskName = (id: string) => ({ 'data_repair': '数据修复', 'anomaly_detection': '异常检测', 'univariate_forecast': '单变量预测', 'multivariate_forecast': '多变量预测', 'covariate_forecast': '协变量预测' }[id] || id)

const requiresMultitaskModel = computed(() => ['data_repair', 'anomaly_detection'].includes(store.currentTask))
const filteredModels = computed(() => store.models.filter(m => {
  if (!m.supported_tasks.includes(getTaskName(store.currentTask))) return false;
  return !requiresMultitaskModel.value || m.name === 'Chronos-2-Multitask';
}))
const selectedModelInfo = computed(() => store.models.find(m => m.name === store.selectedModel))

type ModelParameter = {
  name: string
  default: string | number
  param_type: 'integer' | 'float' | string
  description: string
}

const displayedParameters = computed(() => {
  if (!selectedModelInfo.value) return [];
  const task = store.currentTask;
  if (task === 'data_repair') return []; 
  if (task === 'anomaly_detection') return selectedModelInfo.value.parameters.filter((p: ModelParameter) => p.name === 'anomaly_threshold');
  return selectedModelInfo.value.parameters.filter((p: ModelParameter) => p.name !== 'anomaly_threshold'); 
});

const updateParamDefault = (param: ModelParameter, value: number | undefined) => {
  if (value !== undefined) param.default = String(value);
}

const applyParameterDefaults = () => {
  store.models.forEach(model => {
    model.parameters?.forEach((param: any) => {
      if (param.name === 'prediction_length') param.default = String(store.appSettings.defaultForecastLength);
      if (param.name === 'anomaly_threshold') param.default = String(store.appSettings.defaultAnomalyThreshold);
      if (param.name === 'context_length') param.default = String(store.appSettings.defaultContextLength);
    });
  });
}

const selectModel = (model: any) => {
  if (model.available === false) {
    ElMessage.warning(`${model.name} 权重尚未部署，请先确认模型目录。`);
    return;
  }
  store.selectedModel = model.name;
}

watch(filteredModels, (models) => {
  if (!models.some(model => model.name === store.selectedModel && model.available !== false)) {
    store.selectedModel = '';
  }
})

watch(() => store.appSettings, () => {
  applyParameterDefaults();
  store.loadModels().then(applyParameterDefaults);
}, { deep: true })

onMounted(async () => {
  await store.loadModels();
  applyParameterDefaults();
  store.selectedModel = '';
})

const executeTask = async () => {
  if (!store.selectedModel) return;
  if (selectedModelInfo.value?.available === false) {
    ElMessage.warning("当前模型权重尚未部署，不能执行计算。"); return;
  }
  if (requiresMultitaskModel.value && store.selectedModel !== 'Chronos-2-Multitask') {
    ElMessage.warning("数据修复和异常检测需要选择 Chronos-2-Multitask。"); return;
  }
  const finalTargets = isMultiTarget.value ? (targetVariables.value as string[]) : [targetVariables.value as string];
  
  if (finalTargets.length === 0 || finalTargets[0] === '') {
    ElMessage.warning("请至少选择一个目标分析参数！"); return;
  }
  
  let dStart = 0; let dEnd = store.currentFile.rows;
  if (['data_repair', 'anomaly_detection'].includes(store.currentTask)) {
      dStart = dataRange.value[0];
      dEnd = dataRange.value[1];
      if (dStart >= dEnd) {
          ElMessage.warning("请选择有效的数据提取范围！"); return;
      }
  }

  let predictionLength = store.appSettings.defaultForecastLength;
  let anomalyThreshold = store.appSettings.defaultAnomalyThreshold;
  let contextLength = store.appSettings.defaultContextLength;
  selectedModelInfo.value?.parameters.forEach((p: ModelParameter) => {
    if (p.name === 'prediction_length') predictionLength = parseInt(p.default.toString());
    if (p.name === 'anomaly_threshold') anomalyThreshold = parseFloat(p.default.toString());
    if (p.name === 'context_length') contextLength = parseInt(p.default.toString());
  });
  
  try {
    store.loading = true;
    await store.runPrediction(store.currentTask, finalTargets, covariateVariables.value, predictionLength, confidenceInterval.value, anomalyThreshold, contextLength, dStart, dEnd);
    router.push('/results'); 
  } catch (err: any) { ElMessage.error(`计算失败: ${err.message || err}`); } finally { store.loading = false; }
}
</script>

<style scoped>
.model-select-page { padding: 12px 20px 20px; max-width: 1180px; margin: 0 auto; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.config-layout { display: flex; flex-direction: column; gap: 18px; }
.status-bar { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; padding: 14px 16px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; }
.status-bar > div { min-width: 0; }
.status-label { display: block; margin-bottom: 4px; color: #64748b; font-size: 0.78rem; }
.status-bar strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #0f172a; }
.section-panel { padding: 18px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
.section-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 16px; }
.section-header h3 { margin: 0 0 4px; color: #0f172a; }
.section-header p { margin: 0; color: #64748b; font-size: 0.88rem; }
.model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.model-card-item { position: relative; border: 2px solid #e2e8f0; border-radius: 8px; padding: 16px; cursor: pointer; background-color: #fff; transition: all 0.2s; }
.model-card-item:hover { border-color: #93c5fd; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
.model-card-item.is-active { border-color: #3b82f6; background-color: #eff6ff; }
.model-card-item.is-disabled { cursor: not-allowed; opacity: 0.58; background-color: #f8fafc; }
.model-card-item.is-disabled:hover { border-color: #e2e8f0; transform: none; box-shadow: none; }
.model-info h4 { margin: 0 0 8px 0; font-size: 1.1rem; } .model-info p { margin: 0 0 12px 0; color: #64748b; font-size: 0.9rem; }
.check-mark { position: absolute; top: 12px; right: 12px; color: #3b82f6; font-weight: bold; font-size: 1.2rem; }
.fade-in { animation: fadeIn 0.3s ease-in-out; } @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.mt-4 { margin-top: 1rem; } .mt-8 { margin-top: 2rem; } .flex { display: flex; } .justify-end { justify-content: flex-end; } .text-xl { font-size: 1.25rem; } .font-bold { font-weight: 700; } .m-0 { margin: 0; }
@media (max-width: 900px) { .status-bar { grid-template-columns: 1fr; } }
</style>
