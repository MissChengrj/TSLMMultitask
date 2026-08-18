<template>
  <div class="results-page">
    <div class="page-title">
      <div>
        <h2>多维分析结果报告</h2>
        <p>汇总本次推理任务、模型输出和关键风险点。</p>
      </div>
      <el-button @click="router.push('/model-select')">返回重新配置</el-button>
    </div>

    <div v-if="store.predictionResults && store.predictionResults.multi_results">
      <div class="summary-grid">
        <div class="summary-item">
          <span>任务类型</span>
          <strong>{{ taskName }}</strong>
        </div>
        <div class="summary-item">
          <span>使用模型</span>
          <strong>{{ store.predictionResults.model_used || store.selectedModel || '-' }}</strong>
        </div>
        <div class="summary-item">
          <span>分析变量</span>
          <strong>{{ store.predictionResults.multi_results.length }} 个</strong>
        </div>
        <div class="summary-item">
          <span>推理设备</span>
          <strong>{{ store.predictionResults.device_used || '-' }}</strong>
        </div>
        <div v-if="store.currentTask === 'anomaly_detection'" class="summary-item">
          <span>重构诊断</span>
          <strong>{{ reconstructionHealthText }}</strong>
        </div>
      </div>

      <div v-if="store.currentTask === 'data_repair'" style="margin-bottom: 24px;">
        <el-alert v-if="totalRepairs === 0" title="数据状态完美，未发现空值或异常跳变！" type="success" show-icon :closable="false" />
        <el-alert v-else :title="`数据修复完成 (修复 ${totalRepairs} 处)`" type="warning" show-icon :closable="false" />
      </div>
      <div v-if="store.currentTask === 'anomaly_detection'" style="margin-bottom: 24px;">
        <el-alert v-if="totalAnomalies === 0" title="设备运行平稳，监控参数均处于安全重构误差阈值内！" type="success" show-icon :closable="false" />
        <el-alert v-else :title="`警报：共突破重构误差阈值发现 ${totalAnomalies} 个异常状态点`" type="error" show-icon :closable="false" />
        <el-alert
          v-if="totalReconstructionGaps > 0"
          :title="`模型原始重构中检测到 ${totalReconstructionGaps} 个非有限输出点，后端已完成有限值兜底，当前图表展示的是可用基准线。`"
          type="warning"
          show-icon
          :closable="false"
          style="margin-top: 12px;"
        />
        <el-alert
          v-else
          title="模型重构输出健康，当前结果未检测到原始重构缺口。"
          type="success"
          show-icon
          :closable="false"
          style="margin-top: 12px;"
        />
      </div>

      <div 
        v-for="index in store.predictionResults.multi_results.length" 
        :key="index" 
        class="chart-panel"
        :class="{ 'chart-panel-large': store.currentTask === 'anomaly_detection' }"
      >
        <div :id="`chart-${index - 1}`" style="width: 100%; height: 100%;"></div>
      </div>
    </div>
    <div v-else>
      <el-empty description="暂无计算结果数据">
        <el-button type="primary" @click="router.push('/model-select')">去执行计算</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, nextTick, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useDataStore } from '../stores/data'
import * as echarts from 'echarts'

const store = useDataStore()
const router = useRouter()

const taskNameMap: Record<string, string> = {
  data_repair: '数据修复',
  anomaly_detection: '异常检测',
  univariate_forecast: '单变量预测',
  multivariate_forecast: '多变量预测',
  covariate_forecast: '协变量预测'
}
const taskName = computed(() => taskNameMap[store.currentTask] || store.currentTask || '-')
const totalRepairs = computed(() => store.currentTask === 'data_repair' ? store.predictionResults?.multi_results?.reduce((sum: number, res: any) => sum + (res.repaired_count || 0), 0) : 0)
const totalAnomalies = computed(() => store.currentTask === 'anomaly_detection' ? store.predictionResults?.multi_results?.reduce((sum: number, res: any) => sum + (res.total_anomalies || 0), 0) : 0)
const totalReconstructionGaps = computed(() => store.currentTask === 'anomaly_detection' ? store.predictionResults?.multi_results?.reduce((sum: number, res: any) => sum + (res.reconstruction_gap_count || 0), 0) : 0)
const reconstructionHealthText = computed(() => {
  if (store.currentTask !== 'anomaly_detection') return '-'
  return totalReconstructionGaps.value > 0 ? `${totalReconstructionGaps.value} 个原始缺口` : '输出完整'
})
const charts: echarts.EChartsType[] = []

const resizeCharts = () => charts.forEach(chart => chart.resize())

const toFiniteOrNull = (value: unknown): number | null => {
  const numericValue = Number(value)
  return Number.isFinite(numericValue) ? numericValue : null
}

const sanitizeLineData = (values: unknown[] = []): Array<number | null> => values.map(toFiniteOrNull)

const fillDisplayGaps = (values: unknown[] = [], fallbackValues: unknown[] = []): Array<number | null> => {
  const sanitized = sanitizeLineData(values)
  if (!sanitized.some(value => value === null)) return sanitized

  const fallback = sanitizeLineData(fallbackValues)
  return sanitized.map((value, index) => value ?? fallback[index] ?? null)
}

onMounted(async () => {
  if (!store.predictionResults || !store.predictionResults.multi_results) return;
  await nextTick();
  
  store.predictionResults.multi_results.forEach((res: any, index: number) => {
    const chartDom = document.getElementById(`chart-${index}`);
    if (!chartDom) return;
    const myChart = echarts.init(chartDom);
    charts.push(myChart);
    let option: any = {};

    if (store.currentTask === 'data_repair') {
      option = {
        title: { text: `[${res.target_name}] 数据修复对比` },
        tooltip: { trigger: 'axis' },
        legend: { top: 30, data: ['原始数据', '修复后数据'] },
        grid: { top: 80, left: '5%', right: '5%', bottom: '15%' }, // 预留底部滑动条空间
        // 核心修复：为数据修复加入全景滑动条
        dataZoom: [
          { type: 'slider', bottom: 5 },
          { type: 'inside' }
        ],
        xAxis: { type: 'category', data: res.timestamps, boundaryGap: false }, 
        yAxis: { type: 'value', scale: true },
        series: [
          { name: '原始数据', type: 'line', data: res.original_data, itemStyle: { color: '#cbd5e1' }, lineStyle: { width: 4 } },
          { name: '修复后数据', type: 'line', data: res.repaired_data, itemStyle: { color: '#10b981' }, lineStyle: { type: 'dashed', width: 2 } }
        ]
      };
    } else if (store.currentTask === 'anomaly_detection') {
      const scatterData = (res.anomalies || []).map((a: any) => [res.timestamps[a.index], a.value]);
      const originalData = sanitizeLineData(res.original_data || []);
      const reconstructedData = fillDisplayGaps(res.reconstructed_data || [], res.original_data || []);
      const anomalyScores = sanitizeLineData(res.anomaly_scores || []);
      const hasRawReconstructionGap = (res.reconstruction_gap_count || 0) > 0;
      option = {
        title: {
          text: `[${res.target_name}] 运行状态与重构误差分析`,
          subtext: hasRawReconstructionGap
            ? `模型原始重构缺口: ${res.reconstruction_gap_count} 个，图表已使用后端兜底基准线`
            : '模型重构输出完整'
        },
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { top: 30, data: ['实际监控值', '模型重构基准线', '触发警报点', '异常得分'] },
        axisPointer: { link: [{ xAxisIndex: 'all' }] },
        grid: [
          { top: 96, left: '5%', right: '5%', height: '42%' },    
          { top: '65%', left: '5%', right: '5%', height: '22%' }   
        ],
        xAxis: [
          { type: 'category', data: res.timestamps, boundaryGap: false, gridIndex: 0 },
          { type: 'category', data: res.timestamps, boundaryGap: false, gridIndex: 1 }
        ],
        yAxis: [
          { type: 'value', name: '参数值', scale: true, gridIndex: 0 },
          { type: 'value', name: '异常得分', splitLine: { show: false }, gridIndex: 1 }
        ],
        dataZoom: [
          { type: 'slider', xAxisIndex: [0, 1], bottom: 5 },
          { type: 'inside', xAxisIndex: [0, 1] }
        ],
        series: [
          { name: '实际监控值', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: originalData, itemStyle: { color: '#3b82f6' }, connectNulls: false, z: 5 },
          { name: '模型重构基准线', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: reconstructedData, itemStyle: { color: '#10b981' }, lineStyle: { type: 'dashed', width: 2 }, connectNulls: true, z: 4 },
          { name: '触发警报点', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: scatterData, itemStyle: { color: '#ef4444' }, symbolSize: 12, z: 10 },
          { name: '异常得分', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: anomalyScores, itemStyle: { color: '#f59e0b' }, areaStyle: { opacity: 0.25 }, symbol: 'none', connectNulls: true, markLine: { data: [{ yAxis: res.threshold, name: '动态判别阈值' }], lineStyle: { color: '#ef4444', type: 'dashed', width: 2 }, label: { formatter: '警报阈值: {c}' } } }
        ]
      };
    } else {
      let xAxisData: Array<string | number> = [];
      let historyLine: Array<number | null> = [];
      let predLine: Array<number | null> = [];
      let lowerBand: Array<number | null> = [];
      let ciBand: Array<number | null> = [];
      if (store.showHistoricalData && res.history_data) {
        xAxisData = [...res.history_timestamps, ...res.timestamps];
        historyLine = [...res.history_data, ...Array(res.timestamps.length).fill(null)];
        const lastVal = res.history_data[res.history_data.length - 1];
        const padArr = Array(res.history_data.length - 1).fill(null);
        predLine = [...padArr, lastVal, ...res.predictions];
        lowerBand = [...padArr, lastVal, ...res.confidence_lower];
        ciBand = [...padArr, 0, ...res.confidence_upper.map((u: number, i: number) => u - res.confidence_lower[i])];
      } else {
        xAxisData = res.timestamps || []; predLine = res.predictions || []; lowerBand = res.confidence_lower || [];
        ciBand = res.confidence_upper ? res.confidence_upper.map((u: number, i: number) => u - res.confidence_lower[i]) : [];
      }
      option = {
        title: { text: `[${res.target_name}] 趋势预测` }, 
        tooltip: { trigger: 'axis' },
        legend: { top: 30, data: ['历史数据', '预测中位数'] },
        grid: { top: 80, left: '5%', right: '5%', bottom: '15%' }, // 预留底部滑动条空间
        // 核心修复：为所有预测任务加入全景滑动条
        dataZoom: [
          { type: 'slider', bottom: 5 },
          { type: 'inside' }
        ],
        xAxis: { type: 'category', data: xAxisData, boundaryGap: false }, 
        yAxis: { type: 'value', scale: true },
        series: [
          { name: '历史数据', type: 'line', data: historyLine, itemStyle: { color: '#3b82f6' } },
          { name: '置信下界', type: 'line', data: lowerBand, stack: 'ci', lineStyle: { opacity: 0 }, symbol: 'none' },
          { name: '置信面', type: 'line', data: ciBand, stack: 'ci', areaStyle: { color: 'rgba(251, 146, 60, 0.25)' }, lineStyle: { opacity: 0 }, symbol: 'none' },
          { name: '预测中位数', type: 'line', data: predLine, itemStyle: { color: '#ea580c' }, lineStyle: { width: 3, type: 'dashed' }, symbol: 'circle' }
        ]
      };
    }
    myChart.setOption(option);
  });
  window.addEventListener('resize', resizeCharts);
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeCharts);
  charts.splice(0).forEach(chart => chart.dispose());
})
</script>

<style scoped>
.results-page {
  padding: 12px 20px 24px;
  max-width: 1440px;
  margin: 0 auto;
}

.page-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin-bottom: 18px;
}

.page-title h2 {
  margin: 0 0 4px;
  color: #0f172a;
}

.page-title p {
  margin: 0;
  color: #64748b;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.summary-item {
  min-width: 0;
  padding: 14px 16px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}

.summary-item span {
  display: block;
  margin-bottom: 6px;
  color: #64748b;
  font-size: 0.78rem;
}

.summary-item strong {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #0f172a;
}

.chart-panel {
  height: 500px;
  margin-bottom: 24px;
  padding: 20px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(15, 23, 42, 0.06);
}

.chart-panel-large {
  height: 600px;
}

@media (max-width: 980px) {
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .page-title {
    align-items: flex-start;
    flex-direction: column;
  }

  .summary-grid {
    grid-template-columns: 1fr;
  }
}
</style>
