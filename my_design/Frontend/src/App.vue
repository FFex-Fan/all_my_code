<template>
  <div id="app" style="top:0;left:0;">

    <NavBar></NavBar>

    <transition name="el-fade-in-linear"
                mode="out-in">
      <router-view id="route" :key="$route.fullPath"></router-view>
    </transition>
  </div>
</template>

<script>
import NavBar from "./components/base_page/NavBar.vue";

export default {
  name: "App",
  components: {
    NavBar
  }
};
</script>

<style scoped>
#route {
  position: relative;
  top: 10px;
}


.footer {
  margin-top: 15px;
  margin-bottom: 10px;
  text-align: center;
  font-size: small;
}
</style>
