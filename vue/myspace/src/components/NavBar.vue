<script>
export default {
  name: "NavBar",

}
</script>

<template>
  <nav class="navbar navbar-expand-lg bg-body-tertiary">
    <div class="container">
      <!-- 从后端渲染改为前段渲染：
        1. <a></a> 标签改为 <router-link></router-link> 标签
        2. 将 href= 改成 :to=
        3. :to 中的内容为 ======> "{name: 'router中定义的名称'}"
      -->
      <router-link class="navbar-brand" :to="{name: 'home'}">My Space</router-link>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarText"
              aria-controls="navbarText" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarText">
        <ul class="navbar-nav me-auto mb-2 mb-lg-0">
          <li class="nav-item">
            <router-link class="nav-link active" aria-current="page" :to="{name: 'home'}">首页</router-link>
          </li>
          <li class="nav-item">
            <router-link class="nav-link" :to="{name: 'user_list'}">用户列表</router-link>
          </li>
          <li class="nav-item">
            <router-link class="nav-link" :to="{name: 'user_activity'}">用户动态</router-link>
          </li>
        </ul>
        <ul class="navbar-nav">
          <li class="nav-item">
            <router-link class="nav-link" :to="{name: 'login'}">登陆</router-link>
          </li>
          <li class="nav-item">
            <router-link class="nav-link" :to="{name: 'register'}">注册</router-link>
          </li>
        </ul>

      </div>
    </div>
  </nav>
</template>

<style scoped>

</style>