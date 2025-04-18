import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import Home from './views/Home.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL), // Add base URL support
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Home
    }
  ]
})

// Create the app instance
const app = createApp(App)

// Add error handling
app.config.errorHandler = (err, vm, info) => {
  console.error('Vue Error:', err)
  console.error('Error Info:', info)
}

// Add router
app.use(router)

// Mount with error catching
try {
  app.mount('#app')
  console.log('Vue app mounted successfully')
} catch (error) {
  console.error('Failed to mount Vue app:', error)
}

