<template>
  <div class="card edit-field">
    <div class="card-body">
      <label for="edit-area" class="form-label">编辑区</label>
      <!-- 将 textarea 中的内容和 content 相绑定，即：textarea 和 content 中的内容相同 -->
      <textarea v-model="content" class="form-control" id="edit-area" rows="3" placeholder="请输入您的内容"></textarea>
      <button @click="add_post" type="button" class="btn btn-outline-primary btn-sm">发布</button>
    </div>
  </div>
</template>

<script>
import {ref} from "vue";
import $ from 'jquery';
import {useStore} from "vuex";

export default {
  name: "U_A_write",

  setup(props, context) {
    let content = ref('');  // 用于获取 textarea 中的内容
    const store = useStore();  // JWT 验证需要 useStore 模块

    const add_post = () => {  // 发布动态 函数
      if (content.value === "") return;  // 内容为空则直接返回

      $.ajax({
        url: "https://app165.acapp.acwing.com.cn/myspace/post/",
        type: "POST",
        data: {
          content: content.value,
        },
        headers: {
          'Authorization': "Bearer " + store.state.user.access,
        },
        success(resp) {
          if (resp.result === 'success') {
            context.emit('add_post', content.value);
            content.value = "";  // 由于 content 由 ref 定义，故 content 中数据的读取和修改都要用 content.value 来做
          }
        }
      })
    };

    return {
      content,
      add_post,
    }
  }

}
</script>

<style scoped>
.edit-field {
  margin-top: 20px;
}

textarea {
  //position: relative;
  //transform: translateY(-15%);
}

button {
  margin-top: 10px;
}

</style>