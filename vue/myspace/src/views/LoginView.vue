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
import {useStore} from "vuex";
import router from "@/router/index";

export default {
  name: 'LoginView',
  components: {
    ContentBase,  // 引入公共代码模版
  },
  setup() {
    const store = useStore();  // 首先获取 store

    // 使用 v-model="username/password" 将 input 中的内容与 username/password 双向绑定起来
    let username = ref('');
    let password = ref('');
    let error_message = ref('');  // 错误信息需要定义为响应式变量, 不需要双向绑定（div 中不需要修改）

    // 提交 函数
    const login = () => {
      error_message.value = "";  // 登陆前清空错误信息
      store.dispatch("login", {  // 调用 user 中 actions 中的内容，并且传入参数
        username: username.value,  // 传入参数
        password: password.value,
        success() {
          // console.log("success");
          router.push({name: "user_list"});  // 成功则跳转到 /user_list 页面
        },
        error() {  // 失败的回调函数
          // console.log("failed");
          error_message.value = "用户名或密码错误";
        },
      });
    };

    return {
      username,
      password,
      error_message,
      login,
    }
  }
}
</script>

<style scoped>
button {
  width: 100%;
}

.error_message {
  color: red;
}
</style>

<!--
    GET & POST 方法：
      1. GET: 参数放在链接中，不安全
      2. POST：参数放在 http body 中，较安全
-->

<!--
    加密验证：
        1. client 想 server 发送 用户名 + 密码
        2. server 向 client 返回 JWT，该 JWT 在 server 本地不存储
        3. 用户 info + server 私钥  =====>  验证信息
        4. 返回的内容： 用户 info + 验证信息  (公钥)
        5. 判断是否合法： 用户再次传递信息时，会将 info & 验证信息(签名) 一起发送到服务器，
           若 info + server 密钥 通过加密函数 =====> 验证信息，即合法，否则不合法
-->