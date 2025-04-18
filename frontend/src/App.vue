<template>
  <div id="app">
    <header>
      <nav>
        <router-link to="/" class="nav-link">Home</router-link>
      </nav>
    </header>
    <main>
      <router-view></router-view>
    </main>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
#app {
  font-family: Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}

header {
  background-color: #f8f9fa;
  padding: 1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

nav {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.nav-link {
  color: #2c3e50;
  text-decoration: none;
  font-weight: bold;
}

.nav-link:hover {
  color: #42b983;
}

main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
}
</style>

