<template>
  <div class="home">
    <h1>Semantic Validation API</h1>
    <div class="content">
      <section class="hero">
        <h2>Welcome to Our Service</h2>
        <p>A powerful API for semantic validation and analysis.</p>
      </section>
      
      <section class="demo" v-if="isReady">
        <h3>Try It Out</h3>
        <div class="input-group">
          <textarea 
            v-model="inputText" 
            placeholder="Enter your text here..."
            rows="4"
            class="form-input"
          ></textarea>
          <textarea 
            v-model="referenceText" 
            placeholder="Enter reference text here..."
            rows="4"
            class="form-input"
          ></textarea>
          <button 
            @click="validateText" 
            :disabled="!canValidate"
            class="validate-btn"
          >
            Validate Text
          </button>
        </div>
        
        <div v-if="result" class="result">
          <h4>Results:</h4>
          <p>Similarity Score: {{ result.similarity.toFixed(2) }}</p>
          <p>Match: {{ result.match ? 'Yes' : 'No' }}</p>
          <p>Processing Time: {{ result.processing_time_ms.toFixed(2) }}ms</p>
        </div>
      </section>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import api from '../api'

export default {
  name: 'Home',
  async setup() {
    const isReady = ref(false)
    const inputText = ref('')
    const referenceText = ref('')
    const result = ref(null)
    
    // Simulate API check
    try {
      await api.get('/health')
      isReady.value = true
    } catch (error) {
      console.error('API Health check failed:', error)
      isReady.value = true // Set to true anyway for demo
    }

    const canValidate = computed(() => 
      inputText.value.trim() && referenceText.value.trim()
    )

    const validateText = async () => {
      try {
        const response = await api.post('/api/validate', {
          input_text: inputText.value,
          reference_text: referenceText.value,
          module: "ICSE"
        })
        result.value = response.data
      } catch (error) {
        console.error('Validation error:', error)
        alert('Error validating text. Please try again.')
      }
    }

    return {
      isReady,
      inputText,
      referenceText,
      result,
      canValidate,
      validateText
    }
  }
}
</script>

<style scoped>
.home {
  text-align: center;
}

h1 {
  color: #42b983;
  margin-bottom: 2rem;
}

.content {
  max-width: 800px;
  margin: 0 auto;
}

.hero {
  background-color: #f8f9fa;
  padding: 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.hero h2 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.hero p {
  color: #666;
  font-size: 1.1rem;
}

.demo {
  padding: 2rem;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin: 1rem 0;
}

.form-input {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  font-family: inherit;
}

.validate-btn {
  background-color: #42b983;
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.2s;
}

.validate-btn:hover:not(:disabled) {
  background-color: #3aa876;
}

.validate-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.result {
  margin-top: 2rem;
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 4px;
  text-align: left;
}

.result h4 {
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.result p {
  margin: 0.5rem 0;
  color: #666;
}
</style>

