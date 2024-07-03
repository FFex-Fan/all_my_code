import axios from "axios";
import {Message} from "element-ui";
import store from './index'

import createPersistedState from 'vuex-persistedstate'


const ModelUser = {
  state: {
    username: "",
    access: "",
    refresh: "",
    is_login: false,
  },
  getters: {},
  mutations: {  // 外部调用 mutations 中的某个名称，需要使用 store.commit("属性名称", 参数)
    updateUser(state, user) {  // 更新用户状态
      state.username = user.username;
      state.access = user.access;
      state.refresh = user.refresh;
      state.is_login = user.is_login;
    },
    // 由于 mutations 不能支持异步，故获取信息需要放在 actions 中
    // 又因为 actions 中不能进行修改操作，故更新操作需要放在 mutations 中

    // 退出登陆
  },

  actions: {  // 外部调用 actions 中的某个名字，需要使用 store.dispatch("属性名称", 参数)
    login(context, data) {
      axios
        .post("http://localhost:8000/login/", {
          username: data.username,
          password: data.password,
        })
        .then(response => {
          if (response.data === "passwordError") {
            Message.error("密码错误");
            return;
          }
          Message({
            message: "Success！",
            type: "success"
          });
          sessionStorage.setItem("username", data.username);
          sessionStorage.setItem("type", response.data.type);
          if (store.state.loginip === "") {
            store.state.loginip = "chrome" // 后台处理，存储其他数据
          }
          axios
            .post("http://localhost:8000/setlogindata/", {
              username: data.username,
              ip: store.state.loginip,
              msg: store.state.logininfo
            })
            .then(response => {
              // console.log("logindata success")
              context.commit("updateUser", {
                ...response,
                username: data.username,
                access: "",
                refresh: "",
                is_login: true,  // 登陆成功，修改登陆状态
              })
              // window.sessionStorage.setItem("is_login", "true");
              data.$router.push({name: 'problem'});
              // console.log(context.state.is_login);
            })
            .catch(() => {
              Message.error("服务器错误！");
              // console.log(JSON.stringify(error.response.data));

              sessionStorage.setItem("username", "");
              sessionStorage.setItem("rating", "");
              sessionStorage.setItem("type", "");
              sessionStorage.setItem("acpro", "");
            });
        })
        .catch(() => {
          Message.error("用户名不存在");
        });
    },

    handleCommand(context, data) {
      if (data.command === "logout") {
        axios
          .get("/logout/")
          .then(response => {
            Message({
              message: "Logout Success!",
              type: "success"
            });

            context.commit("updateUser", {
              username: "",
              access: "",
              refresh: "",
              is_login: false,
            });

            sessionStorage.setItem("username", "");
            sessionStorage.setItem("name", "");
            sessionStorage.setItem("rating", "");
            sessionStorage.setItem("type", "");
            data.$router.go(0);
          })
          .catch(error => {
            Message.error("服务器错误！");
            console.log(JSON.stringify(error.response));
          });
      }
      if (data.command === "home") {
        data.$router.push({
          name: "user",
          query: {username: sessionStorage.username}
        });
      }
      if (data.command === "setting") {
        data.$router.push({
          name: "setting",
          params: {username: sessionStorage.username}
        });
      }
      if (data.command === "submission") {
        data.$router.push({
          name: "statue",
          query: {username: sessionStorage.username}
        });
      }
      if (data.command === "admin") {
        data.$router.push({
          name: "admin"
        });
      }
      if (data.command === "classes") {
        data.$router.push({
          name: "classes"
        });
      }
    },
  },
  modules: {},
}

export default ModelUser;
