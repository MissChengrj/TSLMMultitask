<template>
  <div class="results-page" style="padding: 20px; max-width: 1200px; margin: 0 auto;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
      <h2>多维分析结果报告</h2>
      <el-button @click="router.push('/model-select')">返回重新配置</el-button>
    </div>

    <div v-if="store.predictionResults && store.predictionResults.multi_results">
      <div v-if="store.currentTask === 'data_repair'" style="margin-bottom: 24px;">
        <el-alert v-if="totalRepairs === 0" title="数据状态完美，未发现空值或异常跳变！" type="success" show-icon :closable="false" />
        <el-alert v-else :title="`数据修复完成 (修复 ${totalRepairs} 处)`" type="warning" show-icon :closable="false" />
      </div>
      <div v-if="store.currentTask === 'anomaly_detection'" style="margin-bottom: 24px;">
        <el-alert v-if="totalAnomalies === 0" title="设备运行平稳，监控参数均处于安全重构误差阈值内！" type="success" show-icon :closable="false" />
        <el-alert v-else :title="`警报：共突破重构误差阈值发现 ${totalAnomalies} 个异常状态点`" type="error" show-icon :closable="false" />
      </div>

      <!-- 调整了基础高度为 500px，为底部的滑动条留出空间 -->
      <div 
        v-for="(res, index) in store.predictionResults.multi_results" 
        :key="index" 
        :style="{ height: store.currentTask === 'anomaly_detection' ? '600px' : '500px', marginBottom: '40px', background: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 12px rgba(0,0,0,0.05)' }"
      >
        <div :id="`chart-${index}`" style="width: 100%; height: 100%;"></div>
      </div>
    </div>
    <div v-else>
      <el-empty description="暂无计算结果数据" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, nextTick, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useDataStore } from '../stores/data'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'

const store = useDataStore()
const router = useRouter()

const totalRepairs = computed(() => store.currentTask === 'data_repair' ? store.predictionResults?.multi_results?.reduce((sum: number, res: any) => sum + (res.repaired_count || 0), 0) : 0)
const totalAnomalies = computed(() => store.currentTask === 'anomaly_detection' ? store.predictionResults?.multi_results?.reduce((sum: number, res: any) => sum + (res.total_anomalies || 0), 0) : 0)

onMounted(async () => {
  if (!store.predictionResults || !store.predictionResults.multi_results) return;
  await nextTick();
  
  store.predictionResults.multi_results.forEach((res: any, index: number) => {
    const chartDom = document.getElementById(`chart-${index}`);
    if (!chartDom) return;
    const myChart = echarts.init(chartDom);
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
      option = {
        title: { text: `[${res.target_name}] 运行状态与重构误差分析` },
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { top: 30, data: ['实际监控值', '模型重构基准线', '触发警报点', '异常得分'] },
        axisPointer: { link: [{ xAxisIndex: 'all' }] },
        grid: [
          { top: 80, left: '5%', right: '5%', height: '45%' },    
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
          { name: '实际监控值', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: res.original_data, itemStyle: { color: '#3b82f6' }, z: 5 },
          { name: '模型重构基准线', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: res.reconstructed_data, itemStyle: { color: '#10b981' }, lineStyle: { type: 'dashed', width: 2 }, z: 4 },
          { name: '触发警报点', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: scatterData, itemStyle: { color: '#ef4444' }, symbolSize: 12, z: 10 },
          { name: '异常得分', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: res.anomaly_scores, itemStyle: { color: '#f59e0b' }, areaStyle: { opacity: 0.25 }, symbol: 'none', markLine: { data: [{ yAxis: res.threshold, name: '动态判别阈值' }], lineStyle: { color: '#ef4444', type: 'dashed', width: 2 }, label: { formatter: '警报阈值: {c}' } } }
        ]
      };
    } else {
      let xAxisData = []; let historyLine = []; let predLine = []; let lowerBand = []; let ciBand = [];
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
})
</script>