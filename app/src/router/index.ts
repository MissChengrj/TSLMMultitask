import { createRouter, createWebHistory } from 'vue-router'

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
      meta: { title: '模型选择' }
    },
    {
      path: '/tasks',
      name: 'Tasks',
      component: () => import('@/views/Tasks.vue'),
      meta: { title: '任务执行' }
    },
    {
      path: '/results',
      name: 'Results',
      component: () => import('@/views/Results.vue'),
      meta: { title: '结果查看' }
    },
    {
      path: '/settings',
      name: 'Settings',
      component: () => import('@/views/Settings.vue'),
      meta: { title: '系统设置' }
    }
  ]
})

export default router