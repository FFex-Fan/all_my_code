<template>
  <NavBar></NavBar>
  <router-view/>
</template>

<script>
import NavBar from "@/components/base_page/NavBar.vue";

export default {
  name: 'App',
  components: {
    NavBar,
  }
}
</script>

<style scoped>

</style>
