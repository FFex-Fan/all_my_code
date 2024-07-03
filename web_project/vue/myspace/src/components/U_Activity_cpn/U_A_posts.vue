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
            <button @click="delete_post(post.id)" v-if="is_me" type="button" class="btn btn-outline-danger btn-sm">删除</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>

import {useStore} from "vuex";
import {computed} from "vue";
import $ from 'jquery';

export default {
  name: "U_A_posts",

  props: {  // 接受父组件传下来的 posts 参数
    parent_posts: {
      type: Object,
      required: true,
    },
    parent_user: {
      type: Object,
      required: true,
    },
  },

  setup(props, context) {
    const store = useStore();
    const is_me = computed(() => store.state.user.id === props.parent_user.id)


    const delete_post = post_id => {
      $.ajax({
        url:"https://app165.acapp.acwing.com.cn/myspace/post/",
        type: "DELETE",
        data: {
          post_id,
        },
        headers: {
          'Authorization': "Bearer " + store.state.user.access,
        },
        success(resp) {
          if (resp.result === "success") {
            context.emit('delete_post', post_id);
          }
        }
      })

    }

    return {
      is_me,
      delete_post,
    }
  }
}
</script>

<style scoped>
.single-post {
  margin-bottom: 10px;
}

button {
  float: right;
}
</style>