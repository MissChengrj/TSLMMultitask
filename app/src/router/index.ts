import { createRouter, createWebHistory } from 'vue-router'
import { useDataStore } from '@/stores/data'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/data-import'
    },
    {
      path: '/data-import',
      name: 'DataImport',
      component: () => import('@/views/DataImport.vue'),
      meta: { title: '数据导入' }
    },
    {
      path: '/model-select',
      name: 'ModelSelect',
      component: () => import('@/views/ModelSelect.vue'),
      meta: { title: '模型配置', requiresData: true, requiresTask: true }
    },
    {
      path: '/tasks',
      name: 'Tasks',
      component: () => import('@/views/Tasks.vue'),
      meta: { title: '分析任务', requiresData: true }
    },
    {
      path: '/results',
      name: 'Results',
      component: () => import('@/views/Results.vue'),
      meta: { title: '分析结果', requiresResult: true }
    },
    {
      path: '/settings',
      name: 'Settings',
      component: () => import('@/views/Settings.vue'),
      meta: { title: '系统设置' }
    }
  ]
})

router.beforeEach((to) => {
  const store = useDataStore()
  if (to.meta.requiresData && !store.hasData) {
    return '/data-import'
  }
  if (to.meta.requiresTask && !store.currentTask) {
    return '/tasks'
  }
  if (to.meta.requiresResult && !store.predictionResults) {
    return store.hasData && store.currentTask ? '/model-select' : '/data-import'
  }
  return true
})

export default router
