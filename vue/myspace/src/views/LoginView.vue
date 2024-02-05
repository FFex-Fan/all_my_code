<template>
  <ContentBase>
    <!-- 子元素组件 -->
    <!-- justify-content-md-center 类可以使 grid 布局的元素居中 -->
    <div class="row justify-content-md-center">
      <div class="col-3">
        <!--
            1. login 函数需要绑定在 form 中，即：@submit='login'
            2. 由于执行完 login() 后仍会重新执行原本的提交事件, 故提交内容会出现一闪而过
            3. 此时需要将 @submit 替换成 @submit.prevent 来阻止默认事件
        -->
        <form @submit.prevent="login">
          <div class="mb-3">
            <label for="username" class="form-label">用户名</label>
            <input v-model="username" type="text" class="form-control" id="username">
          </div>
          <div class="mb-3">
            <label for="password" class="form-label">密码</label>
            <input v-model="password" type="password" class="form-control" id="password">
          </div>
          <div class="error_message">{{ error_message }}</div>
          <button type="submit" class="btn btn-primary">登陆</button>
        </form>
      </div>
    </div>

  </ContentBase>
</template>

<script>
// @ is an alias to /src
import ContentBase from "@/components/ContentBase.vue";  // import 的名称与 ContentBase 定义的名称可以不同
import {ref} from "vue";

export default {
  name: 'LoginView',
  components: {
    ContentBase,  // 引入公共代码模版
  },
  setup() {
    // 使用 v-model="username/password" 将 input 中的内容与 username/password 双向绑定起来
    let username = ref('');
    let password = ref('');
    let error_message = ref('');  // 错误信息需要定义为响应式变量, 不需要双向绑定（div 中不需要修改）

    // 提交 函数
    const login = () => {
      console.log(username.value, password.value);
    }

    return {
      username,
      password,
      error_message,
      login,
    }
  }
}
</script>

<style>
button {
  width: 100%;
}

.error_message {
  color: red;
}
</style>
