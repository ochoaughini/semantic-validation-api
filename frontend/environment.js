// environment.js
const environment = {
    apiUrl: window.location.hostname === "localhost"
        ? "http://localhost:8080"  // Local development backend
        : "https://semantic-validation-api.onrender.com"  // Render deployment
};

// Allow override through window.ENV
if (window.ENV && window.ENV.apiUrl) {
    environment.apiUrl = window.ENV.apiUrl;
}

// Log the environment configuration
console.log("Environment Configuration:", {
    hostname: window.location.hostname,
    apiUrl: environment.apiUrl
});

export default environment;
