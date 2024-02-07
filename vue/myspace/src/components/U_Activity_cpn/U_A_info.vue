<template>
  <div class="card">
    <div class="card-body">
      <div class="row">
        <div class="col-3  img-field">
          <img class="img-fluid" :src="parent_user.photo" alt="">
        </div>
        <div class="col-9">
          <div class="name">{{ parent_user.username }}</div>
          <div class="fans">粉丝数：{{ parent_user.followerCount }}</div>
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
import $ from 'jquery';
import {useStore} from "vuex";

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

    const store = useStore();

    // +关注 函数（注意：不能在子组件中修改父组件中定义的属性）
    const follow = () => {
      $.ajax({
        url: "https://app165.acapp.acwing.com.cn/myspace/follow/",
        type: "POST",
        data: {
          target_id: props.parent_user.id,
        },
        headers: {
          'Authorization': 'Bearer ' + store.state.user.access,
        },
        success(resp) {
          if (resp.result === 'success') {
            context.emit('follow');  // 通过 context.emit() 方法调用父组件中传递过来的函数
          }
        }
      })

    }

    const unfollow = () => {
      $.ajax({
        url: "https://app165.acapp.acwing.com.cn/myspace/follow/",
        type: "POST",
        data: {
          target_id: props.parent_user.id,
        },
        headers: {
          'Authorization': 'Bearer ' + store.state.user.access,
        },
        success(resp) {
          if (resp.result === 'success') {
            context.emit('unfollow1');  // 通过 context.emit() 方法调用父组件中传递过来的函数
          }

        }
      })

    }

    return {
      follow,
      unfollow,
    }
  }


}
</script>

<style scoped>

.img-fluid {
  border-radius: 50%;
}

.img-field {
  display: flex;
  flex-direction: column;
  justify-content: center;
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