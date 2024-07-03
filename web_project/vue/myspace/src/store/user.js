import $ from 'jquery';
import {jwtDecode} from "jwt-decode";

const ModelUser = {
    state: {
        id: "",
        username: "",
        photo: "",
        followerCount: 0,
        access: "",
        refresh: "",
        is_login: false,
    }, 
    getters: {
        
    },
    mutations: {  // 外部调用 mutations 中的某个名称，需要使用 store.commit("属性名称", 参数)
        updateUser(state, user) {  // 更新用户状态
            state.id = user.id;
            state.username = user.username;
            state.photo = user.photo;
            state.followerCount = user.followerCount;
            state.access = user.access;
            state.refresh = user.refresh;
            state.is_login = user.is_login;
        },

        // 由于 mutations 不能支持异步，故获取信息需要放在 actions 中
        // 又因为 actions 中不能进行修改操作，故更新操作需要放在 mutations 中
        updateAccess(state, access) {  // 更新 access
            state.access = access;
        },

        // 退出登陆
        logout(state) {
            state.id = "";
            state.username = "";
            state.photo = "";
            state.followerCount = 0;
            state.access = "";
            state.refresh = "";
            state.is_login = false;
        }
    },

    actions: {  // 外部调用 actions 中的某个名字，需要使用 store.dispatch("属性名称", 参数)
        // ajax 获取 userid， userid 获取 用户信息
        login(context, data) {
            $.ajax({
                url: "https://app165.acapp.acwing.com.cn/api/token/",
                type: "POST",
                data: {
                    username: data.username,
                    password: data.password,
                },
                success(resp) {  // 成功的回调函数
                    const {access, refresh} = resp;
                    // 获取用户信息，需要用户 id，用户 id 在 access 的 token 中, 故需对 access 进行解码
                    const access_obj = jwtDecode(access)

                    setInterval(() => {  // 定时获取 access (4.5min)
                        $.ajax({
                            url: "https://app165.acapp.acwing.com.cn/api/token/refresh/",
                            type: "POST",
                            data: {
                                refresh,
                            },
                            success(resp) {  // 更新 access
                                context.commit("updateAccess", resp.access)
                            },
                        })
                    }, 1000 * 60 * 4.5)

                    $.ajax({
                        url: "https://app165.acapp.acwing.com.cn/myspace/getinfo/",
                        type: "GET",
                        data: {
                            user_id: access_obj.user_id,
                        },
                        headers: {  // 加上 JWT 验证，若某个 api 没有授权，则不能加上
                            'Authorization': "Bearer " + access,  // Bearer 后面有个空格
                        },
                        success(resp) {
                            context.commit("updateUser", {
                                ...resp,
                                access: access,
                                refresh: refresh,
                                is_login: true,  // 登陆成功，修改登陆状态
                            });
                            console.log("before success");
                            data.success(); // 调用 loginView 传过来成功的回调函数，console 输出 success
                        }
                    })
                },
                error() {
                    data.error(); // 调用 loginView 传过来失败的回调函数，console 输出 failed
                }
            });
        },
    },
    modules: {},
}

export default ModelUser;