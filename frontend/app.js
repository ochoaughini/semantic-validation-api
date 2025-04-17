console.log("Semantic Validator loaded");

document.getElementById("validateButton").addEventListener("click", async () => {
  const inputText = document.getElementById("inputText").value;
  const referenceText = document.getElementById("referenceText").value;
  const resultDiv = document.getElementById("result");

  resultDiv.innerHTML = "Loading...";

  try {
    const response = await fetch("https://semantic-validation-api.onrender.com/api/validate", {
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
    resultDiv.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
  }
});
