import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { invoke } from '@tauri-apps/api/core'

export const useDataStore = defineStore('data', () => {
  const savedSettings = (() => {
    try {
      return JSON.parse(localStorage.getItem('tslm_app_settings') || '{}')
    } catch {
      return {}
    }
  })()
  const currentFile = ref<any>(null)
  const selectedColumns = ref<string[]>([])
  const selectedModel = ref<string>('')
  const models = ref<any[]>([])
  const predictionResults = ref<any>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const currentTask = ref<string>('')
  const showHistoricalData = ref(true)
  const appSettings = ref({
    defaultForecastLength: 24,
    defaultAnomalyThreshold: 3.0,
    defaultRepairMethod: 'model_reconstruction',
    defaultContextLength: 512,
    previewRows: 100,
    autoSaveResults: false,
    pythonPath: 'python',
    modelDir: 'weights',
    enableCpuQuantization: true,
    ...savedSettings
  })
  
  // 核心：默认数据类型为基线数据
  const dataType = ref<string>('baseline') 
  
  const hasData = computed(() => currentFile.value !== null)

  watch(appSettings, (value) => {
    localStorage.setItem('tslm_app_settings', JSON.stringify(value))
  }, { deep: true })

  async function loadModels() {
    try { models.value = await invoke('get_models', { modelDir: appSettings.value.modelDir }) } 
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
        contextLength, dataStart, dataEnd, dataType: dataType.value,
        pythonPath: appSettings.value.pythonPath,
        modelDir: appSettings.value.modelDir,
        enableCpuQuantization: appSettings.value.enableCpuQuantization
      })
      predictionResults.value = res; return res;
    } catch (err: any) { error.value = err; throw err; }
  }

  return { 
    currentFile, selectedColumns, selectedModel, models, predictionResults, 
    loading, error, currentTask, showHistoricalData, appSettings, dataType, hasData, 
    loadModels, loadFileData, runPrediction 
  }
})
