import Vue from 'vue'
import Vuex from 'vuex'
import ModelUser from "./user";

Vue.use(Vuex);
export default new Vuex.Store({
  state: {  // 存储所有数据
    loginip: "后台获取",
    logininfo: "后台获取",
  },
  getters: {  // 需要通过一定的计算来获取 state 中的内容时使用，但不能修改信息，如：通过 firstname 和 lastname 计算全名

  },
  mutations: {  // 不支持异步操作，对于某个属性的修改需要放在 mutations 中
  },
  actions: {  // 一个完整复杂的修改可以放在 action 中

  },
  modules: {  // 将 state 进行分割
    user: ModelUser,
  }
});

// 访问数据(username)： store.state.user.username

