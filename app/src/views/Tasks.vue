<template>
  <div class="tasks-container">
    <el-card class="box-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <h2 class="text-xl font-bold m-0">配置分析任务</h2>
          <span class="text-gray-500 text-sm mt-1">请选择当前发动机 QAR 数据需要执行的下游任务</span>
        </div>
      </template>

      <div class="context-banner mt-2 mb-6" v-if="dataStore.currentFile">
        <div class="flex items-center">
          <span class="status-dot"></span>
          <span class="text-gray-700 font-medium mr-2">当前挂载数据:</span>
          <el-tag type="info" size="large" class="font-mono">
            {{ dataStore.currentFile.filename }}
          </el-tag>
          <span class="text-gray-500 ml-4 text-sm">
            已就绪 ({{ dataStore.currentFile.rows }} 行 × {{ dataStore.currentFile.columns }} 特征)
          </span>
        </div>
      </div>

      <div class="task-grid mt-4">
        <div 
          v-for="task in taskList" 
          :key="task.id"
          class="task-card"
          :class="{ 'is-selected': selectedTaskId === task.id }"
          @click="selectTask(task.id)"
        >
          <div class="task-icon">
            <span class="text-2xl">{{ task.icon }}</span>
          </div>
          <div class="task-content">
            <h3 class="task-title">{{ task.name }}</h3>
            <p class="task-desc">{{ task.description }}</p>
          </div>
          <div class="check-mark" v-if="selectedTaskId === task.id">
            ✓
          </div>
        </div>
      </div>

      <div class="action-footer mt-8 flex justify-between">
        <el-button size="large" @click="goBack">
          返回上一步
        </el-button>
        <el-button 
          type="primary" 
          size="large" 
          :disabled="!selectedTaskId" 
          @click="goToNextStep"
        >
          下一步：选择模型配置
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';
// 引入全局状态管理，以便获取刚才上传的文件信息并保存选择的任务
import { useDataStore } from '../stores/data'; 

const router = useRouter();
const dataStore = useDataStore();

// 当前选中的任务 ID
const selectedTaskId = ref<string>('');

// 任务列表配置 (严格按照您的需求)
const taskList = [
  {
    id: 'data_repair',
    name: '数据修复',
    icon: '🔧',
    description: '针对 QAR 数据的缺失值、传感器噪声及异常尖峰进行平滑与插值修复，提升数据质量。'
  },
  {
    id: 'anomaly_detection',
    name: '异常检测',
    icon: '🚨',
    description: '通过重构误差或时序特征分析，智能定位发动机运行过程中的异常状态与潜在故障点。'
  },
  {
    id: 'univariate_forecast',
    name: '单变量预测',
    icon: '📈',
    description: '针对单一核心性能指标（如排气温度 EGT、燃油流量 FF）进行未来趋势与剩余寿命预测。'
  },
  {
    id: 'multivariate_forecast',
    name: '多变量预测',
    icon: '📊',
    description: '同时输入并预测多个关联特征（如 EGT、N1、N2），捕捉复杂系统内部参数的相互作用。'
  },
  {
    id: 'covariate_forecast',
    name: '协变量预测',
    icon: '🌍',
    description: '引入已知的外部条件或工况标签（如高度、马赫数、Op Code）作为已知协变量，辅助提升目标参数的预测精度。'
  }
];

// 选择任务的方法
const selectTask = (id: string) => {
  selectedTaskId.value = id;
};

// 返回上一步
const goBack = () => {
  router.push('/import'); // 根据您的路由配置，可能是 '/' 或 '/import'
};

// 提交任务并进入下一步
const goToNextStep = () => {
  if (!selectedTaskId.value) return;
  
  // 将选中的任务类型存入 Pinia 全局状态中
  dataStore.$patch({
    currentTask: selectedTaskId.value
  });
  
  // 跳转到模型选择或参数配置页面
  router.push('/model-select'); 
};
</script>

<style scoped>
.tasks-container {
  padding: 20px;
  max-width: 1000px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  flex-direction: column;
}

/* 上下文提示条 */
.context-banner {
  background-color: #f8fafc;
  border-left: 4px solid #3b82f6;
  padding: 12px 16px;
  border-radius: 4px;
}

.status-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  background-color: #10b981;
  border-radius: 50%;
  margin-right: 10px;
  box-shadow: 0 0 0 2px #d1fae5;
}

/* 任务卡片网格布局 */
.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

/* 卡片样式 */
.task-card {
  position: relative;
  display: flex;
  align-items: flex-start;
  padding: 20px;
  background-color: #ffffff;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.task-card:hover {
  border-color: #93c5fd;
  transform: translateY(-2px);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.task-card.is-selected {
  border-color: #2563eb;
  background-color: #eff6ff;
}

.task-icon {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f1f5f9;
  border-radius: 8px;
  margin-right: 16px;
}

.task-card.is-selected .task-icon {
  background-color: #dbeafe;
}

.task-content {
  flex: 1;
}

.task-title {
  margin: 0 0 8px 0;
  font-size: 1.1rem;
  color: #1e293b;
  font-weight: 600;
}

.task-desc {
  margin: 0;
  font-size: 0.85rem;
  color: #64748b;
  line-height: 1.5;
}

/* 选中标记 */
.check-mark {
  position: absolute;
  top: 16px;
  right: 16px;
  color: #2563eb;
  font-size: 1.2rem;
  font-weight: bold;
}

/* 工具类 */
.mt-1 { margin-top: 0.25rem; }
.mt-2 { margin-top: 0.5rem; }
.mt-4 { margin-top: 1rem; }
.mt-6 { margin-top: 1.5rem; }
.mt-8 { margin-top: 2rem; }
.mb-6 { margin-bottom: 1.5rem; }
.mr-2 { margin-right: 0.5rem; }
.ml-4 { margin-left: 1rem; }
.flex { display: flex; }
.items-center { align-items: center; }
.justify-between { justify-content: space-between; }
.text-sm { font-size: 0.875rem; }
.text-xl { font-size: 1.25rem; }
.text-2xl { font-size: 1.5rem; }
.font-bold { font-weight: 700; }
.font-medium { font-weight: 500; }
.font-mono { font-family: monospace; }
.text-gray-500 { color: #6b7280; }
.text-gray-700 { color: #374151; }
</style>