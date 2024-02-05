<template>
    <NavBar></NavBar>


  <router-view/>
</template>

<script>
import NavBar from "@/components/NavBar.vue";

export default {
  name: "App",
  components: {  // 在 components 中注册后，在 template 中才可以使用
    NavBar,  // 此处为简写形式，完整格式为 ======>  NavBar: NavBar
  }
}
</script>

<style scoped>

</style>
