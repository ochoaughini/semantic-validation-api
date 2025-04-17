// environment.js
const environment = {
    apiUrl: window.location.hostname === "localhost"
        ? "http://localhost:8080"  // Local development backend
        : process.env.NODE_ENV === "development"
            ? "https://semantic-validation-api-dev.vercel.app"  // Development environment
            : "https://semantic-validation-api.vercel.app"  // Production environment
};

// Allow override through window.ENV
if (window.ENV && window.ENV.apiUrl) {
    environment.apiUrl = window.ENV.apiUrl;
}

// Log the environment configuration in non-production environments
if (process.env.NODE_ENV !== "production") {
    console.log("Environment Configuration:", {
        hostname: window.location.hostname,
        apiUrl: environment.apiUrl,
        nodeEnv: process.env.NODE_ENV
    });
}

export default environment;
