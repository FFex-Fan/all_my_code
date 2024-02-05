<!-- 用于存放抽象的出公共代码模版 -->
<template>
  <div class="container">
    <div class="card">
      <div class="card-body">
        <!-- slot 可以获取所有子元素，然后渲染出来 -->
        <slot></slot>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "ContentBase",

}
</script>

<style scoped>
.container {
  margin-top: 20px;
}
</style>