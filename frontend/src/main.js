import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import Home from './views/Home.vue'
import config from './config'

console.log('Starting application with config:', config);

const router = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            name: 'Home',
            component: Home
        }
    ]
})

const app = createApp(App)

// Add error handling
app.config.errorHandler = (err, vm, info) => {
    console.error('Vue Error:', err)
    console.error('Error Info:', info)
}

// Add configuration
app.config.globalProperties.$config = config

// Add router
app.use(router)

// Mount with error catching
try {
    app.mount('#app')
    console.log('Vue app mounted successfully')
} catch (error) {
    console.error('Failed to mount Vue app:', error)
}

