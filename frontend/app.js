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

        const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                input_text: inputText,
                reference_text: referenceText,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail?.[0]?.msg || "Validation failed");
        }

        const data = await response.json();
        resultDiv.innerHTML = `
            <p><strong>Input:</strong> ${data.input}</p>
            <p><strong>Reference:</strong> ${data.reference}</p>
            <p><strong>Match:</strong> ${data.match ? "✅ Yes" : "❌ No"}</p>
        `;
    } catch (error) {
        console.error("API Error:", error);
        resultDiv.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    }
});
