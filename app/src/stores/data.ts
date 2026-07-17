import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { invoke } from '@tauri-apps/api/core'

export const useDataStore = defineStore('data', () => {
  const currentFile = ref<any>(null)
  const selectedColumns = ref<string[]>([])
  const selectedModel = ref<string>('')
  const models = ref<any[]>([])
  const predictionResults = ref<any>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const currentTask = ref<string>('')
  const showHistoricalData = ref(true)
  
  // 核心：默认数据类型为基线数据
  const dataType = ref<string>('baseline') 
  
  const hasData = computed(() => currentFile.value !== null)

  async function loadModels() {
    try { models.value = await invoke('get_models') } 
    catch (err: any) { error.value = err }
  }

  // 核心：读取文件时强制带入 dataType
  async function loadFileData(path: string, isExcel: boolean) {
    try {
      const command = isExcel ? 'read_excel_file' : 'read_csv_file';
      currentFile.value = await invoke(command, { path: path, dataType: dataType.value });
      return currentFile.value;
    } catch (err: any) { throw err; }
  }

  // 核心：预测时强制带入 dataType
  async function runPrediction(
    task: string, columns: string[], covariates: string[], 
    forecastLength: number, confidenceInterval: number, anomalyThreshold: number,
    contextLength: number, dataStart: number, dataEnd: number
  ) {
    try {
      const res = await invoke('run_prediction', {
        path: currentFile.value.path, modelName: selectedModel.value, predictionType: task,
        columns, covariates, forecastLength, confidenceInterval, anomalyThreshold,
        contextLength, dataStart, dataEnd, dataType: dataType.value
      })
      predictionResults.value = res; return res;
    } catch (err: any) { error.value = err; throw err; }
  }

  return { 
    currentFile, selectedColumns, selectedModel, models, predictionResults, 
    loading, error, currentTask, showHistoricalData, dataType, hasData, 
    loadModels, loadFileData, runPrediction 
  }
})