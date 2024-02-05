<template>
  <div class="card">
    <div class="card-body">
      <div class="row">
        <div class="col-3">
          <img class="img-fluid" src="https://cdn.acwing.com/media/user/profile/photo/236181_lg_2b756ca6cf.webp" alt="">
        </div>
        <div class="col-9">
          <div class="name">{{ fullName }}</div>
          <div class="fans">粉丝数：{{ parent_user.fans_cnt }}</div>
          <!--
              1. v-on:click  <====>  @click  ，用于将点击事件和对应函数绑定起来
              2. v-if   ，如果条件成立，则显示该标签，否则不显示该标签
          -->
          <button
              v-on:click="follow"
              v-if="!parent_user.is_followed"
              type="button"
              class="btn btn-primary btn-sm"
          >
            +关注
          </button>
          <button
              @click="unfollow"
              v-if="parent_user.is_followed"
              type="button"
              class="btn btn-secondary btn-sm"
          >
            取消关注
          </button>
        </div>
      </div>
    </div>
  </div>

</template>

<script>
import {computed} from "vue";

export default {
  name: "U_A_info",

  // 父组件向子组件传递参数
  props: {
    parent_user: {  // 必填
      type: Object,
      required: true,
    },
  },

  // setup() 有两个参数，props 和 context
  setup(props, context) {
    // 计算出对应的字符串
    let fullName = computed(() => props.parent_user.last_name + " " + props.parent_user.first_name);


    // +关注 函数（注意：不能在子组件中修改父组件中定义的属性）
    const follow = () => {
      context.emit('follow');  // 通过 context.emit() 方法调用父组件中传递过来的函数
    }

    const unfollow = () => {
      context.emit('unfollow1');  // 通过 context.emit() 方法调用父组件中传递过来的函数
    }

    return {
      fullName,
      follow,
      unfollow,
    }
  }


}
</script>

<style scoped>

.img-fluid {
  border-radius: 50%;
  margin: 0 auto;
  position: relative;
  top: 50%;
  transform: translateY(-50%);

}

.name {
  font-weight: bold;
}

.fans {
  margin-top: 3px;
  font-size: 12px;
  color: grey;
}

button {
  margin-top: 3px;
  padding: 2px 4px;
}

</style>