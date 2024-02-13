<template>
  <ContentBase>
    <!-- 子元素组件 -->
    <div class="row">
      <div class="col-3">
        <!--
            父组件向子组件传递信息：
                1. 在子组件标签中添加参数  ====>  :（传递到子组件中，子组件的调用名称）="父组件参数名称,也可以是一个表达式"
                2. :parent_user="user"  <====>  v-bind:parent_user="user"
                3. 若想要在某一属性中使用变量而不是字符串，在改属性前加上 : （v-bind）即可
                4. @follow/unfollow 向 子组件info 中传递父组件定义的函数，使在 子组件info 中可以调用父组件的函数，修改对应属性
                5. @(父组件中定义的名称，即：子组件中调用的名称)="父组件中定义的函数名"
        -->
        <U_A_info :parent_user="user" @follow="follow" @unfollow1="unfollow"></U_A_info>
        <U_A_write v-if="is_me" @add_post="add_post"></U_A_write>
      </div>
      <div class="col-9">
        <U_A_posts :parent_user="user" :parent_posts="posts" @delete_post="delete_post"></U_A_posts>
      </div>
    </div>
  </ContentBase>
</template>

<script>
// @ is an alias to /src
import ContentBase from "@/components/ContentBase.vue";  // import 的名称与 ContentBase 定义的名称可以不同

import U_A_info from "@/components/U_Activity_cpn/U_A_info.vue";
import U_A_write from "@/components/U_Activity_cpn/U_A_write.vue";
import U_A_posts from "@/components/U_Activity_cpn/U_A_posts.vue";

import {computed, reactive} from "vue";
import {useRoute} from "vue-router";
import $ from 'jquery';
import {useStore} from "vuex";

export default {
  name: 'UserActivityView',
  components: {
    ContentBase,  // 引入公共代码模版
    U_A_info,
    U_A_posts,
    U_A_write,
  },
  setup() {
    const store = useStore();
    const route = useRoute();  // 获取链接信息
    const user_id = parseInt(route.params.user_id);  // 获取链接中的 user_id 属性
    // 改属性的值需要在 U_A_info 中被渲染出来，即：需要在不同组件之间传递信息
    const user = reactive({});
    // 定义动态列表中的数据
    const posts = reactive({});

    // 判断是否是自己的页面
    const is_me = computed(() => user_id === store.state.user.id);

    $.ajax({
      url: "https://app165.acapp.acwing.com.cn/myspace/getinfo/",
      type: "GET",
      data: {
        user_id: user_id,
      },
      headers: {
        'Authorization': "Bearer " + store.state.user.access,
      },
      success(resp) {
        user.id = resp.id;
        user.username = resp.username;
        user.photo = resp.photo;
        user.followerCount = resp.followerCount;
        user.is_followed = resp.is_followed;
      }
    });

    $.ajax({
      url: "https://app165.acapp.acwing.com.cn/myspace/post/",
      type: "GET",
      data: {
        user_id: user_id,
      },
      headers: {
        'Authorization': "Bearer " + store.state.user.access,
      },
      success(resp) {
        posts.count = resp.length;
        posts.posts = resp;
      },
    })



    // +关注， 父组件中定义的参数必须在父组件中才能进行修改
    const follow = () => {
      if (user.is_followed) return;
      user.followerCount++;
      user.is_followed = true;
    }

    // 取消关注
    const unfollow = () => {
      if (!user.is_followed) return;
      user.followerCount--;
      user.is_followed = false;
    }

    // 发布动态
    const add_post = (content) => {
      posts.count++;
      posts.posts.unshift({  // unshift() 将内容加入到最上面，push() 将内容加入到最下面
        id: posts.count,
        user_id: 1,
        content: content,
      })
    }

    // 删除动态
    const delete_post = post_id => {
      posts.posts = posts.posts.filter(post => post.id !== post_id);
      posts.count = posts.posts.length;
    }

    return {
      user,
      posts,
      follow,
      unfollow,
      add_post,
      is_me,
      delete_post,
    }
  }
}
</script>

<style>

</style>
