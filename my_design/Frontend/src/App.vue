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
  },
  created() {
    if (sessionStorage.getItem('store')) {  // 恢复本地 sessionStorage 数据
      this.$store.replaceState(
        Object.assign(
          {},
          this.$store.state,
          JSON.parse(sessionStorage.getItem('store'))
        )
      )
    }
    // 在页面刷新时将vuex里的信息保存到 sessionStorage 里,beforeunload事件在页面刷新时先触发
    window.addEventListener('beforeunload', () => {
      sessionStorage.setItem('store', JSON.stringify(this.$store.state))
    })
  }
};
</script>

<style scoped>
#route {
  position: relative;
  top: 10px;
  width: 100%;
  height: 100%;
}
.footer {
  margin-top: 15px;
  margin-bottom: 10px;
  text-align: center;
  font-size: small;
}
</style>
