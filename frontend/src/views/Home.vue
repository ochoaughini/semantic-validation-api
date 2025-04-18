<template>
  <div class="home">
    <h1>Semantic Validation API</h1>
    <div class="content">
      <section class="hero">
        <h2>Welcome to Our Service</h2>
        <p>A powerful API for semantic validation and analysis.</p>
      </section>
      
      <section class="features">
        <h3>Features</h3>
        <ul>
          <li>Semantic Text Analysis</li>
          <li>Medical Terminology Validation</li>
          <li>Content Similarity Checking</li>
          <li>Advanced NLP Processing</li>
        </ul>
      </section>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Home'
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

.features {
  text-align: left;
  padding: 1rem;
}

.features h3 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.features ul {
  list-style: none;
  padding: 0;
}

.features li {
  padding: 0.5rem 0;
  color: #666;
  position: relative;
  padding-left: 1.5rem;
}

.features li::before {
  content: "✓";
  color: #42b983;
  position: absolute;
  left: 0;
}
</style>

