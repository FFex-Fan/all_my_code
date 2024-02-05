<template>
  <ContentBase>
    <!-- 子元素组件 -->
    Register
  </ContentBase>
</template>

<script>
// @ is an alias to /src
import ContentBase from "@/components/ContentBase.vue";  // import 的名称与 ContentBase 定义的名称可以不同

export default {
  name: 'RegisterView',
  components: {
    ContentBase,  // 引入公共代码模版
  }
}
</script>

<style>

</style>
