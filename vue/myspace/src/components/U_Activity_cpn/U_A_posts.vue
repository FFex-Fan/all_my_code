<template>
  <div class="card">
    <div class="card-body">
      <!--
          1. v-for: 类似于 Python 中的循环
          2. 若使用 v-for，则需要额外绑定一个 :key 属性（一般可以使用 id 作为值，但不推荐使用下标作为 key）
          3. 若使用 v-for="(post, idx) in parent_posts.posts", 则同时可以将下标 idx 返回回来
      -->
      <div v-for="post in parent_posts.posts" :key="post.id">
        <div class="card single-post">
          <div class="card-body">
            {{ post.content }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "U_A_posts",

  props: {  // 接受父组件传下来的 posts 参数
    parent_posts: {
      type: Object,
      required: true,
    },
  }
}
</script>

<style scoped>
.single-post {
  margin-bottom: 10px;
}
</style>