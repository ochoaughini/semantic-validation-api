import config from "./config.js";

console.log("Semantic Validator loaded");
console.log("API Configuration:", config);

document.getElementById("validateButton").addEventListener("click", async () => {
    const inputText = document.getElementById("inputText").value;
    const referenceText = document.getElementById("referenceText").value;
    const resultDiv = document.getElementById("result");

    if (!inputText.trim() || !referenceText.trim()) {
        resultDiv.innerHTML = "<p style=\"color:red;\">Error: Both input and reference texts are required.</p>";
        return;
    }

    resultDiv.innerHTML = "Loading...";

    try {
        const apiUrl = `${config.apiBaseUrl}${config.apiEndpoint}`;
        console.log("Making request to:", apiUrl);
        
        const requestBody = {
            input_text: inputText,
            reference_text: referenceText,
        };
        console.log("Request payload:", requestBody);

        // Attempt to fetch with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
        
        try {
            const response = await fetch(apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(requestBody),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            console.log("Response status:", response.status);
            console.log("Response headers:", Object.fromEntries([...response.headers]));
            
            if (!response.ok) {
                let errorMessage = `Server error (${response.status})`;
                try {
                    const errorData = await response.json();
                    console.error("Server error details:", errorData);
                    errorMessage = errorData.detail?.[0]?.msg || errorMessage;
                } catch (jsonError) {
                    console.error("Failed to parse error response:", jsonError);
                    // Try to get text if JSON parsing fails
                    const errorText = await response.text();
                    console.log("Error response text:", errorText);
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            console.log("Successful response:", data);
            
            resultDiv.innerHTML = `
                <p><strong>Input:</strong> ${data.input}</p>
                <p><strong>Reference:</strong> ${data.reference}</p>
                <p><strong>Match:</strong> ${data.match ? "✅ Yes" : "❌ No"}</p>
            `;
        } finally {
            clearTimeout(timeoutId);
        }
    } catch (error) {
        console.error("API Error:", error);
        
        let errorMessage = error.message;
        if (error.name === 'AbortError') {
            errorMessage = "Request timed out. The server took too long to respond.";
        } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorMessage = "Network error. Unable to connect to the validation service.";
        }
        
        console.error({
            type: error.name,
            message: error.message,
            apiUrl: `${config.apiBaseUrl}${config.apiEndpoint}`,
            stack: error.stack
        });
        
        resultDiv.innerHTML = `
            <div style="color:red;">
                <p><strong>Error:</strong> ${errorMessage}</p>
                <p><small>API URL: ${config.apiBaseUrl}${config.apiEndpoint}</small></p>
                <p><small>If this error persists, please check your network connection or contact support.</small></p>
            </div>
        `;
    }
});
