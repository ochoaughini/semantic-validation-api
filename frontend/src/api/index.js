import axios from 'axios';
import config from '../config';

// Create axios instance with default config
const api = axios.create({
    baseURL: config.apiUrl,
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    },
    // Ensure credentials are sent
    withCredentials: true
});

// Add request interceptor for error handling
api.interceptors.request.use(
    (config) => {
        // Log the request for debugging
        console.log('API Request:', {
            url: config.url,
            method: config.method,
            baseURL: config.baseURL
        });
        return config;
    },
    (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
    }
);

// Add response interceptor for error handling
api.interceptors.response.use(
    (response) => {
        // Log successful responses for debugging
        console.log('API Response:', {
            status: response.status,
            url: response.config.url
        });
        return response;
    },
    (error) => {
        // Log errors for debugging
        console.error('API Error:', {
            message: error.message,
            response: error.response,
            config: error.config
        });

        // Handle specific error cases
        if (error.response) {
            // Server responded with error status
            switch (error.response.status) {
                case 401:
                    console.error('Unauthorized access');
                    break;
                case 403:
                    console.error('Forbidden access');
                    break;
                case 404:
                    console.error('Resource not found');
                    break;
                case 500:
                    console.error('Server error');
                    break;
                default:
                    console.error('API error:', error.response.status);
            }
        } else if (error.request) {
            // Request made but no response received
            console.error('No response received:', error.request);
        }

        return Promise.reject(error);
    }
);

export default api;

