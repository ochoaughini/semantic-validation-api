// src/config.js
const config = {
    apiUrl: import.meta.env.VITE_API_URL || (
        window.location.hostname === "localhost"
            ? "http://localhost:8080"
            : "https://semantic-validation-api.onrender.com"
    )
};

export default config;

