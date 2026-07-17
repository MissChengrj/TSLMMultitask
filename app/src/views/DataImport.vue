<template>
  <div class="data-import-container">
    <el-card class="box-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <h2 class="text-xl font-bold m-0">导入本地 QAR / 监控数据</h2>
          <span class="text-gray-500 text-sm mt-1">数据将在本地进行解析，确保绝对保密</span>
        </div>
      </template>

      <!-- ================= 新增：在选择文件前，先选择数据格式 ================= -->
      <div class="data-type-selector mb-6" style="padding: 20px; background: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1;">
        <h3 style="margin-top: 0; margin-bottom: 16px; color: #1e293b;">第一步：选择数据源格式</h3>
        <el-radio-group v-model="dataStore.dataType">
          <el-radio value="baseline" size="large" border>
            标准基线数据 
            <span style="font-size: 12px; color: #909399; margin-left: 8px;">(含 Flight DateTime/Tail，特征自动从第5列提取)</span>
          </el-radio>
          <el-radio value="qar" size="large" border>
            航空 QAR 数据 
            <span style="font-size: 12px; color: #909399; margin-left: 8px;">(含 FRAME_COUNTER，特征自动从第4列提取)</span>
          </el-radio>
        </el-radio-group>
      </div>
      <!-- ====================================================================== -->

      <div class="upload-area">
        <h3 style="margin-top: 0; margin-bottom: 16px; color: #1e293b;">第二步：导入本地文件</h3>
        <el-button type="primary" size="large" :loading="isLoading" @click="handleSelectFile" class="import-btn">
          {{ isLoading ? '数据解析中，请稍候...' : '选择本地数据文件 (.csv, .xlsx, .xls)' }}
        </el-button>
        <div class="el-upload__tip mt-2">支持标准的 .csv 或 Excel 格式文件。全程离线计算。</div>
      </div>

      <el-alert v-if="errorMessage" :title="errorMessage" type="error" show-icon class="mt-4" @close="errorMessage = ''" />

      <div v-if="fileInfo" class="file-info-section mt-6 transition-fade">
        <h3 class="section-title">📁 文件加载成功</h3>
        
        <el-descriptions :column="3" border class="mt-4" direction="vertical">
          <el-descriptions-item label="发动机序列号 (ESN)">
            <el-tag size="large" type="primary" effect="dark">{{ fileInfo.engine_sn || '解析失败' }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="机尾号 (Tail)">
            <el-tag size="large" type="info">{{ fileInfo.tail || '未知' }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="发位 (ENG POS)">{{ fileInfo.eng_pos || '-' }}</el-descriptions-item>
          <el-descriptions-item label="飞行时间范围">
            <span class="text-gray-600 font-mono">{{ fileInfo.start_time || '未知' }} — {{ fileInfo.end_time || '未知' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="运行代码 (Op Code)">{{ fileInfo.op_code || '-' }}</el-descriptions-item>
          <el-descriptions-item label="有效特征数据">
            <span class="text-green-600 font-bold">{{ fileInfo.rows }}</span> 行 × <span class="text-blue-600 font-bold">{{ fileInfo.columns }}</span> 特征
          </el-descriptions-item>
        </el-descriptions>

        <div class="columns-preview mt-6">
          <h4 class="mb-3 text-gray-700">检测到的特征列 (已剥离元数据)：</h4>
          <div class="tags-container">
            <el-tag v-for="(col, index) in fileInfo.column_names.slice(0, 20)" :key="index" class="m-1" effect="plain">{{ col }}</el-tag>
            <el-tag v-if="fileInfo.column_names.length > 20" type="info" class="m-1">...等共 {{ fileInfo.column_names.length }} 列</el-tag>
          </div>
        </div>

        <div class="action-footer mt-8 flex justify-end">
          <el-button type="success" size="large" @click="goToNextStep">下一步：选择分析任务</el-button>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { useDataStore } from '../stores/data';

const router = useRouter();
const dataStore = useDataStore();
const isLoading = ref(false);
const errorMessage = ref('');
const fileInfo = ref<any>(null);

const handleSelectFile = async () => {
  errorMessage.value = '';
  try {
    const selected = await open({
      multiple: false, filters: [{ name: 'Data', extensions: ['csv', 'xlsx', 'xls'] }]
    });
    if (!selected) return;
    
    const selectedPath = Array.isArray(selected) ? selected[0] : selected;
    isLoading.value = true;
    
    const lowerPath = selectedPath.toLowerCase();
    let result;
    
    // ================= 核心修复：在这里将 dataStore.dataType 传给后端 =================
    if (lowerPath.endsWith('.csv')) {
      result = await invoke('read_csv_file', { 
        path: selectedPath,
        dataType: dataStore.dataType 
      });
    } else {
      result = await invoke('read_excel_file', { 
        path: selectedPath,
        dataType: dataStore.dataType 
      });
    }
    // ==============================================================================

    fileInfo.value = result;
    dataStore.$patch({ currentFile: result, selectedColumns: result.column_names });
  } catch (error: any) {
    errorMessage.value = error.message || error;
  } finally {
    isLoading.value = false;
  }
};

const goToNextStep = () => router.push('/tasks');
</script>

<style scoped>
.data-import-container { padding: 20px; max-width: 1000px; margin: 0 auto; }
.card-header { display: flex; flex-direction: column; }
.upload-area { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 8px; transition: all 0.3s; }
.upload-area:hover { border-color: #409eff; background-color: #ecf5ff; }
.section-title { margin-top: 0; margin-bottom: 16px; color: #1f2937; font-size: 1.125rem; font-weight: 600; border-bottom: 1px solid #e5e7eb; padding-bottom: 10px; }
.tags-container { display: flex; flex-wrap: wrap; }
.transition-fade { animation: fadeIn 0.4s ease-in-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.mt-1 { margin-top: 0.25rem; } .mt-2 { margin-top: 0.5rem; } .mt-4 { margin-top: 1rem; } .mt-6 { margin-top: 1.5rem; } .mt-8 { margin-top: 2rem; } .mb-3 { margin-bottom: 0.75rem; } .mb-6 { margin-bottom: 1.5rem; } .m-0 { margin: 0; } .m-1 { margin: 0.25rem; } .flex { display: flex; } .justify-end { justify-content: flex-end; } .text-gray-500 { color: #6b7280; } .text-gray-700 { color: #374151; } .text-sm { font-size: 0.875rem; } .text-xl { font-size: 1.25rem; } .font-bold { font-weight: 700; }
</style>