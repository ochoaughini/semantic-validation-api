import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import Home from './views/Home.vue'
import config from './config'

// Log startup information
console.log('Starting application with config:', config);

// Create router with proper base URL
const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL || '/'),
    routes: [
        {
            path: '/',
            name: 'Home',
            component: Home
        },
        // Catch-all route for 404
        {
            path: '/:pathMatch(.*)*',
            redirect: '/'
        }
    ]
})

// Create Vue app
const app = createApp(App)

// Global error handler
app.config.errorHandler = (err, vm, info) => {
    console.error('Vue Error:', err)
    console.error('Error Info:', info)
    console.error('Component:', vm?.$options?.name || 'Unknown')
}

// Global warning handler
app.config.warnHandler = (msg, vm, trace) => {
    console.warn('Vue Warning:', msg)
    console.warn('Trace:', trace)
}

// Add configuration to global properties
app.config.globalProperties.$config = config

// Router error handling
router.onError((error) => {
    console.error('Router error:', error)
})

// Add router to app
app.use(router)

// Mount app with comprehensive error handling
const mountApp = async () => {
    try {
        // Wait for router to be ready
        await router.isReady()
        
        // Mount the app
        app.mount('#app')
        console.log('Vue app mounted successfully')
        
        // Log successful initialization
        console.log('Application initialized successfully')
    } catch (error) {
        console.error('Critical error mounting app:', error)
        
        // Show error to user
        const rootDiv = document.getElementById('app')
        if (rootDiv) {
            rootDiv.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                    <h1>Error Starting Application</h1>
                    <p>Please refresh the page or try again later.</p>
                </div>
            `
        }
    }
}

// Start the application
mountApp()

