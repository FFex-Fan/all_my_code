<template>
  <ContentBase>
    <!-- 子元素组件 -->
    <div class="card" v-for="user in users" :key="user.id">
      <div class="card-body">
        <div class="row">
          <div class="col-1">
            <img class="img-fluid" :src="user.photo" alt="">
          </div>
          <div class="col-11">
            <div class="username">{{ user.username }}</div>
            <div class="fans-count">粉丝数: {{ user.followerCount }}</div>
          </div>
        </div>
      </div>
    </div>
  </ContentBase>
</template>

<script>
// @ is an alias to /src
import ContentBase from "@/components/ContentBase.vue";  // import 的名称与 ContentBase 定义的名称可以不同
import $ from 'jquery';  // 使用 Ajax 需要先饮用 jquery 中的 $
import {ref} from "vue";

export default {
  name: 'UserListView',
  components: {
    ContentBase,  // 引入公共代码模版
  },

  setup() {
    let users = ref([]);
    $.ajax({
      url: 'https://app165.acapp.acwing.com.cn/myspace/userlist/',
      type: 'GET',
      success(resp) {
        console.log(resp);
        users.value = resp;  // 将请求到的数据放到 users 变量中存储
      }
    });

    return {
      users,
    };
  }


}
</script>

<style>
img {
  border-radius: 50%;
}

.username {
  font-weight: bold;
  height: 50%;
}

.fans-count {
  font-size: 12px;
  color: grey;
  margin-top: 3px;
  height: 50%;
}

.card {
  margin-bottom: 20px;
  cursor: pointer; /* 将鼠标变成小手的形状 */
}

.card:hover {
  box-shadow: 2px 2px 10px lightgrey; /* 阴影效果 */
  transition: 500ms; /* 降低速度，使该动画持续 0.5s */
}
</style>
